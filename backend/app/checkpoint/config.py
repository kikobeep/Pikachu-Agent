"""Agent Run Checkpoint 的领域模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from app.model.config import Message, MessageRole, ToolCall
from app.agent.result import AgentStopReason
from app.tools.config import ToolResult
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CheckpointStatus(StrEnum):
    """Checkpoint 的持久化生命周期。"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CheckpointPhase(StrEnum):
    """Run 最后确认到的执行边界。"""

    STARTING = "starting"
    MODEL_REQUEST = "model_request"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULTS_READY = "tool_results_ready"
    FINISHED = "finished"


class RunCheckpoint(BaseModel):
    """一次 Run 的最小可恢复状态，不保存完整聊天历史。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    conversation_id: str | None = None
    user_message: Message
    status: CheckpointStatus
    phase: CheckpointPhase
    step: int = Field(default=0, ge=0)
    pending_tool_calls: tuple[ToolCall, ...] = ()
    completed_tool_results: tuple[ToolResult, ...] = ()
    stop_reason: AgentStopReason | None = None
    error: str | None = None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    recovered_by_run_id: str | None = None
    revision: int = Field(default=1, ge=1)

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, value: Message) -> Message:
        if value.role is not MessageRole.USER:
            raise ValueError("checkpoint user_message must have user role")
        return value

    @field_validator("run_id", "conversation_id", "recovered_by_run_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("checkpoint identifiers cannot be empty")
        return normalized

    @field_validator("started_at", "updated_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint datetimes must include timezone information")
        return value.astimezone(UTC)


__all__ = [
    "CheckpointPhase",
    "CheckpointStatus",
    "RunCheckpoint",
]
