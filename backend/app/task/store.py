
from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.task.config import Task, TaskPatch, TaskStatus, TaskStep, TaskStepStatus

_CONTEXT_TASK_STATUSES = frozenset({TaskStatus.ACTIVE, TaskStatus.PAUSED})
DEFAULT_TASKS_DIR = Path(__file__).resolve().parents[2] / ".database" / "tasks"

_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class TaskStore:
    def __init__(self, tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> None:
        self.tasks_dir = Path(tasks_dir).expanduser().resolve()
        self._locks: dict[str, asyncio.Lock] = {}


    async def initialize(self):
        await asyncio.to_thread(self.tasks_dir.mkdir, parents=True, exist_ok=True)

    async def _all_tasks(self) -> list[Task]:
        """读取 tasks 目录中所有持久化任务，按更新时间倒序。"""

        def _read() -> list[Task]:
            tasks: list[Task] = []
            for path in sorted(self.tasks_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                try:
                    tasks.append(Task.model_validate(payload))
                except ValueError:
                    continue
            tasks.sort(key=lambda task: task.updated_at, reverse=True)
            return tasks

        return await asyncio.to_thread(_read)


    async def task_for_conversation(
        self,
        conversation_id: str,
    ) -> Task | None:


        normalized = _normalize(
            conversation_id,
            field_name="conversation_id",
        )
        tasks = [
            task
            for task in await self._all_tasks()
            if normalized == task.owner_conversation_id
            and task.status in _CONTEXT_TASK_STATUSES
        ]
        tasks.sort(key=lambda task: task.updated_at, reverse=True)
        return tasks[0] if tasks else None

    async def create(
        self,
        *,
        title: str,
        description: str | None,
        goal: str | None,
        priority,
        steps,
        owner_conversation_id: str,
        run_ids,
    ) -> Task:
        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        now = datetime.now(UTC)
        task = Task(
            id=uuid4().hex,
            title=title,
            description=description,
            goal=goal,
            priority=priority,
            steps=tuple(steps),
            owner_conversation_id=normalized_owner,
            run_ids=tuple(run_ids),
            created_at=now,
            updated_at=now,
        )
        await asyncio.to_thread(
            _write_task,
            self.tasks_dir / f"{task.id}.json",
            task,
        )
        return task

    async def apply_patch(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        owner_conversation_id: str,
    ) -> Task:
        task = await self.resolve(
            task_id,
            owner_conversation_id=owner_conversation_id,
        )
        if task is None:
            raise KeyError(f"任务不存在：{task_id}")

        if (
            patch.expected_revision is not None
            and patch.expected_revision != task.revision
        ):
            raise ValueError(
                f"任务版本冲突：期望 {patch.expected_revision}，"
                f"当前 {task.revision}"
            )

        data = task.model_dump()
        fields = patch.model_fields_set

        if "goal" in fields:
            data["goal"] = patch.goal
        if patch.status is not None:
            data["status"] = patch.status
        if "state" in fields:
            data["state"] = patch.state or ()
        if patch.add_constraints:
            data["constraints"] = (*task.constraints, *patch.add_constraints)
        if patch.add_key_facts:
            data["key_facts"] = (*task.key_facts, *patch.add_key_facts)
        if "replace_steps" in fields:
            data["steps"] = patch.replace_steps or ()
        if patch.step_id is not None:
            data["steps"] = _update_step(
                task.steps,
                patch.step_id,
                patch.step_status,
                patch.step_note,
            )
        if patch.run_id is not None and patch.run_id not in task.run_ids:
            data["run_ids"] = (*task.run_ids, patch.run_id)

        data["updated_at"] = datetime.now(UTC)
        data["revision"] = task.revision + 1
        if patch.status in _TERMINAL_STATUSES:
            data["completed_at"] = data["updated_at"]
        elif patch.status is not None:
            data["completed_at"] = None

        updated = Task.model_validate(data)
        await asyncio.to_thread(
            _write_task,
            self.tasks_dir / f"{updated.id}.json",
            updated,
        )
        return updated

    async def list(
        self,
        *,
        limit: int,
        status: TaskStatus | None,
        owner_conversation_id: str,
    ) -> list[Task]:
        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        tasks = [
            task
            for task in await self._all_tasks()
            if task.owner_conversation_id == normalized_owner
        ]
        if status is not None:
            tasks = [task for task in tasks if task.status is status]
        return tasks[:limit]

    async def resolve(
        self,
        task_id: str,
        *,
        owner_conversation_id: str,
    ) -> Task | None:
        """按完整 ID 或唯一前缀解析任务；模糊时抛错，不存在返回 None。"""

        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        prefix = task_id.strip().lower()
        if not prefix:
            return None
        matches = [
            task
            for task in await self._all_tasks()
            if task.owner_conversation_id == normalized_owner
            and task.id.startswith(prefix)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"任务 ID 前缀不唯一：{task_id}")
        return matches[0]

    async def active_for_conversation(
        self,
        conversation_id: str,
    ) -> Task | None:
        return await self.task_for_conversation(conversation_id)


def _update_step(
    steps: tuple[TaskStep, ...],
    step_id: str,
    step_status: TaskStepStatus | None,
    step_note: str | None,
) -> tuple[TaskStep, ...]:
    """就地更新单个步骤的状态与备注。"""

    updated: list[TaskStep] = []
    found = False
    for step in steps:
        if step.id == step_id:
            found = True
            data = step.model_dump()
            if step_status is not None:
                data["status"] = step_status
            if step_note is not None:
                data["note"] = step_note
            updated.append(TaskStep.model_validate(data))
        else:
            updated.append(step)
    if not found:
        raise ValueError(f"步骤不存在：{step_id}")
    return tuple(updated)


def _write_task(path: Path, task: Task) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        task.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized
