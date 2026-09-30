"""任务管理工具：供主模型在长任务中创建、更新、查询任务。

任务状态独立于会话消息持久化，不受上下文压缩影响。工具把 Task 领域能力
暴露给模型，让模型能够：

- ``task_create``：为一个整体目标创建任务，并把目标内的子工作拆成步骤；
- ``task_update``：步骤完成、状态变化、补充约束/事实或关联执行记录时更新任务；
- ``task_get``：需要重新确认单个任务当前状态时获取详情；
- ``task_list``：需要总览任务（含用户明确要求列出）时获取列表。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.model.config import AgentMode
from app.task.config import (
    Task,
    TaskPatch,
    TaskPriority,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
)
from app.tools.config import BaseTool, ToolDefinition
from app.tools.register import ToolRegistry

from ..tools.config import ToolExecutionContext
from .store import TaskStore

_MAX_LIST_LIMIT = 100


class TaskCreateTool(BaseTool):
    """创建用于长期跟踪进度的任务。"""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_create",
            record_output=False,
            description=(
                "创建一个任务用于长期跟踪一个整体目标。当工作复杂需要拆解、跨多轮"
                "跟踪，或用户明确要求记录任务时调用。同一整体目标中的阶段、模块和"
                "动作应放入本任务的 steps，不要分别创建多个任务；只有彼此独立、可"
                "分别完成和关闭的目标才创建多个任务。任务状态独立于对话保存，不会"
                "因上下文压缩而丢失。priority 只能是 low、normal、high、urgent。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "任务标题，简洁描述要完成的工作。",
                    },
                    "description": {
                        "type": "string",
                        "description": "可选的任务详细描述。",
                    },
                    "goal": {
                        "type": "string",
                        "description": "可选的当前目标，说明任务要达成的结果。",
                    },
                    "priority": {
                        "type": "string",
                        "enum": [p.value for p in TaskPriority],
                        "description": "可选的任务优先级，默认 normal。",
                    },
                    "steps": {
                        "type": "array",
                        "description": "可选的任务步骤拆解。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "步骤标题。",
                                },
                                "note": {
                                    "type": "string",
                                    "description": "可选的步骤说明。",
                                },
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            strict=False,
            closing_allowed=True,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation_id(context)
        title = arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")

        description = arguments.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("'description' must be a string")
        goal = arguments.get("goal")
        if goal is not None and not isinstance(goal, str):
            raise ValueError("'goal' must be a string")

        priority = TaskPriority.NORMAL
        raw_priority = arguments.get("priority")
        if raw_priority is not None:
            priority = TaskPriority(raw_priority)

        steps = _build_steps(arguments.get("steps"))

        task = await self._store.create(
            title=title,
            description=description,
            goal=goal,
            priority=priority,
            steps=steps,
            owner_conversation_id=conversation_id,
            run_ids=(
                (context.run_id,)
                if context is not None and context.run_id
                else ()
            ),
        )
        return task.model_dump(mode="json")


class TaskUpdateTool(BaseTool):
    """更新任务：推进步骤、改变状态、补充约束/事实或关联执行记录。"""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_update",
            record_output=False,
            description=(
                "更新一个已有任务。当某个步骤完成、任务状态变化、需要补充用户"
                "约束或关键事实时调用。当前会话和运行由系统自动关联；至少提供 "
                "task_id 与一个更新字段。更新系统注入的当前活动任务时，优先把 "
                "task_id 设为 current，避免转录长 ID；其他任务仍使用精确 ID 或"
                "唯一前缀。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "任务 ID、唯一前缀，或当前活动任务句柄 current。"
                        ),
                    },
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in TaskStatus],
                        "description": (
                            "新的任务状态；当步骤 blocked（等待用户输入或外部"
                            "条件）时，可把任务置为 paused，使下次恢复时模型"
                            "明确知道在等什么。"
                        ),
                    },
                    "goal": {
                        "type": "string",
                        "description": "替换任务的当前目标。",
                    },
                    "state": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "替换为最新的当前状态事实。",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "追加的用户约束（去重）。",
                    },
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "追加的关键事实或决策（去重）。",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "可选的新步骤计划；提供时整体替换当前步骤。已有步骤应"
                            "保留原 id，新步骤可省略 id 由系统生成。"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": [s.value for s in TaskStepStatus],
                                },
                                "note": {"type": "string"},
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                    "step_id": {
                        "type": "string",
                        "description": "要推进的步骤 ID；与 step_status 配合。",
                    },
                    "step_status": {
                        "type": "string",
                        "enum": [s.value for s in TaskStepStatus],
                        "description": (
                            "步骤的新状态；标记为 done 时必须同时提供 "
                            "step_note 记录完成依据，标记为 blocked 时必须"
                            "提供 step_note 说明阻塞原因。"
                        ),
                    },
                    "step_note": {
                        "type": "string",
                        "description": (
                            "步骤备注；标记为 done 时记录完成依据，标记为 "
                            "blocked 时说明阻塞原因，两者均必填。"
                        ),
                    },
                    "expected_revision": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "可选的期望任务版本；与当前版本不一致时拒绝覆盖。"
                        ),
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            strict=False,
            closing_allowed=True,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("'task_id' must be a non-empty string")

        has_update = any(
            key in arguments
            for key in (
                "status",
                "goal",
                "state",
                "constraints",
                "facts",
                "steps",
                "step_id",
                "step_status",
            )
        )
        if not has_update:
            raise ValueError(
                "task_update requires at least one update field besides task_id"
            )

        if context is not None and context.mode is AgentMode.PLAN:
            if "status" in arguments:
                raise ValueError(
                    "plan mode 下不能直接改变任务状态；任务由用户接受后才开始"
                )
            if (
                arguments.get("step_id") is not None
                or arguments.get("step_status") is not None
            ):
                raise ValueError(
                    "plan mode 下不能推进步骤状态；只允许更新计划内容"
                )

        # 先验证全部字段，再执行一次原子写入。
        step_id = arguments.get("step_id")
        step_status = arguments.get("step_status")
        if step_id is not None or step_status is not None:
            if step_id is None or step_status is None:
                raise ValueError(
                    "'step_id' and 'step_status' must be provided together"
                )
        if "steps" in arguments and (
            step_id is not None
            or step_status is not None
            or "step_note" in arguments
        ):
            raise ValueError(
                "'steps' cannot be combined with step_id/step_status/step_note"
            )
        goal: str | None = None
        if "goal" in arguments:
            goal = arguments["goal"]
            if goal is not None and not isinstance(goal, str):
                raise ValueError("'goal' must be a string")
        state: tuple[str, ...] | None = None
        if "state" in arguments:
            raw_state = arguments["state"]
            if not isinstance(raw_state, list) or not all(
                isinstance(item, str) for item in raw_state
            ):
                raise ValueError("'state' must be a list of strings")
            state = tuple(raw_state)
        constraints: tuple[str, ...] = ()
        if "constraints" in arguments:
            raw_constraints = arguments["constraints"]
            if not isinstance(raw_constraints, list) or not all(
                isinstance(item, str) for item in raw_constraints
            ):
                raise ValueError("'constraints' must be a list of strings")
            constraints = tuple(raw_constraints)
        facts: tuple[str, ...] = ()
        if "facts" in arguments:
            raw_facts = arguments["facts"]
            if not isinstance(raw_facts, list) or not all(
                isinstance(item, str) for item in raw_facts
            ):
                raise ValueError("'facts' must be a list of strings")
            facts = tuple(raw_facts)
        replacement_steps: tuple[TaskStep, ...] | None = None
        if "steps" in arguments:
            replacement_steps = _build_update_steps(arguments["steps"])
        status = (
            TaskStatus(arguments["status"])
            if "status" in arguments
            else None
        )
        expected_revision = arguments.get("expected_revision")
        if expected_revision is not None and (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
        ):
            raise ValueError("'expected_revision' must be a positive integer")
        step_note = arguments.get("step_note")
        if step_note is not None and not isinstance(step_note, str):
            raise ValueError("'step_note' must be a string")
        if step_status in ("done", "blocked") and (
            step_note is None or not step_note.strip()
        ):
            if step_status == "done":
                raise ValueError(
                    "将步骤标记为 done 时必须提供 step_note 说明完成依据"
                )
            raise ValueError(
                "将步骤标记为 blocked 时必须提供 step_note 说明阻塞原因"
            )

        conversation_id = _require_conversation_id(context)
        task = await _resolve_owned(self._store, task_id, conversation_id)

        patch_data: dict[str, Any] = {
            "status": status,
            "add_constraints": constraints,
            "add_key_facts": facts,
            "step_id": step_id,
            "step_status": (
                TaskStepStatus(step_status) if step_status is not None else None
            ),
            "expected_revision": expected_revision,
            "run_id": context.run_id if context is not None else None,
        }
        if "goal" in arguments:
            patch_data["goal"] = goal
        if "state" in arguments:
            patch_data["state"] = state
        if "steps" in arguments:
            patch_data["replace_steps"] = replacement_steps
        if "step_note" in arguments:
            patch_data["step_note"] = step_note

        task = await self._store.apply_patch(
            task.id,
            TaskPatch.model_validate(patch_data),
            owner_conversation_id=conversation_id,
        )

        return task.model_dump(mode="json")


class TaskGetTool(BaseTool):
    """获取单个任务的完整详情，用于重新确认当前状态。"""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_get",
            record_output=False,
            description=(
                "获取一个任务的完整详情（目标、状态、步骤、约束、关键事实与关联"
                "记录）。当模型需要重新确认当前活动任务时，优先使用 current；"
                "其他任务使用精确 ID 或唯一前缀。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "任务 ID、唯一前缀，或当前活动任务句柄 current。"
                        ),
                    }
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("'task_id' must be a non-empty string")
        conversation_id = _require_conversation_id(context)
        task = await _resolve_owned(self._store, task_id, conversation_id)
        return task.model_dump(mode="json")


class TaskListTool(BaseTool):
    """列出任务，可按状态过滤。"""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_list",
            record_output=False,
            description=(
                "列出任务（可按状态过滤）。当需要总览当前有哪些任务、查看进度，"
                "或用户明确要求列出任务时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in TaskStatus],
                        "description": "可选的状态过滤。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_LIST_LIMIT,
                        "description": (
                            f"返回数量上限，默认 50，最大 {_MAX_LIST_LIMIT}。"
                        ),
                    },
                },
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute(arguments, context=None)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return await self._execute(arguments, context=context)

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        limit = arguments.get("limit", 50)
        if not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(
                f"'limit' must be an integer between 1 and {_MAX_LIST_LIMIT}"
            )

        status: TaskStatus | None = None
        raw_status = arguments.get("status")
        if raw_status is not None:
            status = TaskStatus(raw_status)

        conversation_id = _require_conversation_id(context)
        tasks = await self._store.list(
            limit=limit,
            status=status,
            owner_conversation_id=conversation_id,
        )
        return {
            "count": len(tasks),
            "tasks": [_task_brief(task) for task in tasks],
        }


def register_task_tools(registry: ToolRegistry, store: TaskStore) -> None:
    """把任务管理工具注册进已有工具注册表。"""

    registry.register(TaskCreateTool(store))
    registry.register(TaskUpdateTool(store))
    registry.register(TaskGetTool(store))
    registry.register(TaskListTool(store))



async def _resolve_owned(
    store: TaskStore,
    task_id: str,
    conversation_id: str,
) -> Task:
    """先按当前会话归属过滤，再解析任务 ID 或唯一前缀。

    任务不存在或不属于当前会话时统一返回“任务不存在”，避免泄露
    其他会话中任务的存在性。
    """

    normalized = task_id.strip()
    if normalized.lower() == "current":
        task = await store.active_for_conversation(conversation_id)
    else:
        task = await store.resolve(
            normalized,
            owner_conversation_id=conversation_id,
        )
    if task is None:
        raise KeyError(f"任务不存在：{task_id}")
    return task


def _require_conversation_id(
    context: ToolExecutionContext | None,
) -> str:
    """模型任务工具必须在一个可识别的会话中执行。"""

    if context is None or not context.conversation_id:
        raise ValueError("task tool requires conversation context")
    return context.conversation_id


def _task_brief(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status.value,
        "priority": task.priority.value,
        "goal": task.goal,
        "progress": task.progress_summary,
        "updated_at": task.updated_at.isoformat(),
    }


def _build_steps(raw_steps: object) -> tuple[TaskStep, ...]:
    if raw_steps is None:
        return ()
    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a list")
    steps: list[TaskStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValueError(f"steps[{index}] must be an object")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"steps[{index}].title must be a non-empty string")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"steps[{index}].note must be a string")
        steps.append(
            TaskStep(
                id=uuid4().hex,
                title=title,
                note=note,
            )
        )
    return tuple(steps)


def _build_update_steps(raw_steps: object) -> tuple[TaskStep, ...]:
    """解析重排后的完整计划，保留已有 ID 并为新增步骤生成 ID。"""

    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a list")
    steps: list[TaskStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValueError(f"steps[{index}] must be an object")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"steps[{index}].title must be a non-empty string")
        step_id = item.get("id") or uuid4().hex
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"steps[{index}].id must be a non-empty string")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"steps[{index}].note must be a string")
        raw_status = item.get("status", TaskStepStatus.TODO.value)
        steps.append(
            TaskStep(
                id=step_id,
                title=title,
                status=TaskStepStatus(raw_status),
                note=note,
            )
        )
    return tuple(steps)


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "register_task_tools",
]
