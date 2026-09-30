
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from app.model.config import AgentMode
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunStatus(StrEnum):
    """Run 的生命周期状态
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.INTERRUPTED: frozenset(),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)


class Run(BaseModel):
    """一次 Agent Run 的生命周期记录（轻量索引，不保存事件/工具结果）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    conversation_id: str | None = None
    status: RunStatus
    user_message: str = ""
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    stop_reason: str | None = None
   
    recovered_from_run_id: str | None = None
    # 触发来源（轻量 provenance，随 Run 持久化）：
    #   source       —— manual | automation
    #   source_id    —— automation_id（或其它来源标识）
    #   scheduled_for / triggered_at —— Automation 调度语义
    source: str | None = None
    source_id: str | None = None
    scheduled_for: datetime | None = None
    triggered_at: datetime | None = None
    # 一次执行的模式（NORMAL / PLAN）。不是生命周期状态，只是输入语义。
    mode: AgentMode = AgentMode.DEFAULT

    @field_validator("id", "conversation_id", "recovered_from_run_id", "source_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("run identifiers cannot be empty")
        return normalized

    @field_validator(
        "created_at",
        "started_at",
        "updated_at",
        "completed_at",
        "scheduled_for",
        "triggered_at",
    )
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = [
    "TERMINAL_STATUSES",
    "Run",
    "RunStatus",
    "_ALLOWED_TRANSITIONS",
]
