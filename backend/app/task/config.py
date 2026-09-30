"""任务领域的核心数据结构。

Task 是任务事实的权威源，独立于会话消息持久化。长任务的目标、约束、
进度、待办与关键事实保存在这里，不会因为对话压缩（工具结果缩短、
旧工具轮移除、滚动摘要）而丢失。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

TASK_ID_LENGTH = 32
_TASK_ID_RE = re.compile(rf"^[0-9a-f]{{{TASK_ID_LENGTH}}}$")
_MAX_TITLE_CHARS = 500
_MAX_TEXT_CHARS = 4_000
_MAX_ENTRY_CHARS = 2_000
_MAX_ENTRIES = 100
_MAX_STEPS = 100


class TaskStatus(StrEnum):
    """任务生命周期状态。"""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(StrEnum):
    """任务优先级。"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStepStatus(StrEnum):
    """任务步骤状态。"""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class TaskStep(BaseModel):
    """任务中的一个可追踪步骤。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    status: TaskStepStatus = TaskStepStatus.TODO
    note: str | None = None

    @field_validator("id", "title", mode="before")
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> str:
        """步骤 ID 和标题必须是规范化后的非空文本。"""

        if not isinstance(value, str):
            raise TypeError("task step id and title must be strings")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("task step id and title cannot be empty")
        maximum = 128 if info.field_name == "id" else _MAX_TITLE_CHARS
        if len(normalized) > maximum:
            raise ValueError("task step id or title is too long")
        return normalized

    @field_validator("note", mode="before")
    @classmethod
    def normalize_note(cls, value: object) -> str | None:
        """步骤备注折叠空白；空串转为 None。"""

        normalized = _normalize_optional_text(value)
        if normalized is not None and len(normalized) > _MAX_TEXT_CHARS:
            raise ValueError("task step note is too long")
        return normalized

    @model_validator(mode="after")
    def validate_status_note(self) -> TaskStep:
        """完成与阻塞步骤必须记录可恢复的依据。"""

        if self.status in {TaskStepStatus.DONE, TaskStepStatus.BLOCKED}:
            if not self.note:
                raise ValueError(
                    f"task step with status {self.status.value} requires a note"
                )
        return self


class Task(BaseModel):
    """一个可恢复、可查询的长任务。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str | None = None
    goal: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    constraints: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    steps: tuple[TaskStep, ...] = ()
    owner_conversation_id: str = Field(min_length=1, frozen=True)
    run_ids: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    revision: int = Field(default=1, ge=1)

    @field_validator("title", "description", "goal", mode="before")
    @classmethod
    def normalize_optional_text(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> str | None:
        """标题、描述与目标折叠空白；空串转 None。"""

        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("text fields must be strings or None")
        normalized = _normalize_text(value)
        maximum = _MAX_TITLE_CHARS if info.field_name == "title" else _MAX_TEXT_CHARS
        if len(normalized) > maximum:
            raise ValueError(f"task {info.field_name} is too long")
        return normalized or None

    @field_validator(
        "constraints",
        "state",
        "key_facts",
        mode="before",
    )
    @classmethod
    def normalize_entries(cls, value: object) -> tuple[str, ...]:
        """移除空项并按首次出现顺序去重。"""

        return _normalize_entries(value)

    @field_validator("title")
    @classmethod
    def title_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("task title cannot be empty")
        return value

    @field_validator("id", mode="before")
    @classmethod
    def id_required(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("task id must be a string")
        normalized = value.strip().lower()
        if not _TASK_ID_RE.fullmatch(normalized):
            raise ValueError("task id must be a 32-character hexadecimal string")
        return normalized

    @field_validator("owner_conversation_id", mode="before")
    @classmethod
    def normalize_owner_conversation_id(cls, value: object) -> str:
        """任务必须永久归属于一个非空会话。"""

        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("task owner_conversation_id cannot be empty")
        return normalized

    @field_validator("created_at", "updated_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        """任务时间必须带时区，并统一为 UTC。"""

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task datetimes must include timezone information")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_invariants(self) -> Task:
        """验证步骤、任务状态与时间顺序。"""

        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("task step ids must be unique")
        if len(self.steps) > _MAX_STEPS:
            raise ValueError(f"task cannot contain more than {_MAX_STEPS} steps")
        in_progress_count = sum(
            step.status is TaskStepStatus.IN_PROGRESS for step in self.steps
        )
        if in_progress_count > 1:
            raise ValueError("task can contain at most one in_progress step")
        if self.status is TaskStatus.PAUSED and in_progress_count:
            raise ValueError("paused task cannot contain an in_progress step")
        if self.status is TaskStatus.COMPLETED and self.steps:
            if any(step.status is not TaskStepStatus.DONE for step in self.steps):
                raise ValueError("completed task requires all steps to be done")
        for field_name in ("constraints", "state", "key_facts"):
            if len(getattr(self, field_name)) > _MAX_ENTRIES:
                raise ValueError(
                    f"task {field_name} cannot contain more than {_MAX_ENTRIES} entries"
                )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.completed_at is not None:
            if self.completed_at < self.created_at:
                raise ValueError("completed_at cannot be earlier than created_at")
        return self

    @property
    def progress_summary(self) -> str:
        """渲染一行的进度摘要，用于列表与日志。"""

        total = len(self.steps)
        done = sum(1 for step in self.steps if step.status is TaskStepStatus.DONE)
        status_text = self.status.value
        if total:
            return f"[{status_text}] {self.title} ({done}/{total} 步骤完成)"
        return f"[{status_text}] {self.title}"


class TaskPatch(BaseModel):
    """一次任务更新的完整变更集，用于原子校验和写入。"""

    model_config = ConfigDict(extra="forbid")

    goal: str | None = None
    status: TaskStatus | None = None
    state: tuple[str, ...] | None = None
    add_constraints: tuple[str, ...] = ()
    add_key_facts: tuple[str, ...] = ()
    replace_steps: tuple[TaskStep, ...] | None = None
    step_id: str | None = None
    step_status: TaskStepStatus | None = None
    step_note: str | None = None
    run_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("goal", "step_note", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        return _normalize_optional_text(value)

    @field_validator("step_id", "run_id", mode="before")
    @classmethod
    def normalize_optional_identifier(cls, value: object) -> str | None:
        normalized = _normalize_optional_text(value)
        return normalized

    @field_validator(
        "state",
        "add_constraints",
        "add_key_facts",
        mode="before",
    )
    @classmethod
    def normalize_patch_entries(cls, value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        return _normalize_entries(value)

    @model_validator(mode="after")
    def validate_step_update(self) -> TaskPatch:
        if (self.step_id is None) != (self.step_status is None):
            raise ValueError("step_id and step_status must be provided together")
        if self.step_note is not None and self.step_id is None:
            raise ValueError("step_note requires step_id and step_status")
        if "replace_steps" in self.model_fields_set and self.step_id is not None:
            raise ValueError(
                "replace_steps cannot be combined with a single step update"
            )
        return self

    @property
    def has_changes(self) -> bool:
        """是否包含 revision 以外的实际更新。"""

        explicit_nullable = bool(
            {"goal", "state", "replace_steps"} & self.model_fields_set
        )
        return explicit_nullable or any(
            (
                self.status is not None,
                bool(self.add_constraints),
                bool(self.add_key_facts),
                self.step_id is not None,
                self.run_id is not None,
            )
        )


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    normalized = _normalize_text(value)
    return normalized or None


def _normalize_entries(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[object] = (value,) if isinstance(value, str) else value
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in values:
        if not isinstance(entry, str):
            raise TypeError("task entries must be strings")
        text = _normalize_text(entry)
        if len(text) > _MAX_ENTRY_CHARS:
            raise ValueError("task entry is too long")
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


__all__ = [
    "Task",
    "TASK_ID_LENGTH",
    "TaskPriority",
    "TaskPatch",
    "TaskStatus",
    "TaskStep",
    "TaskStepStatus",
]
