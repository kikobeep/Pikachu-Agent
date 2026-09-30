"""Agent 执行过程的统一事件模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from app.model.config import Message, ToolCall
from app.model.config import ModelUsage
from app.tools.approval import ApprovalDecision
from app.tools.config import ToolResult
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .result import AgentError, AgentResult, AgentStopReason


class AgentEventType(StrEnum):
    """Agent 生命周期中可产生的事件类型。"""

    AGENT_STARTED = "agent_started"
    MODEL_STARTED = "model_started"
    MODEL_OUTPUT_DELTA = "model_output_delta"
    MODEL_REASONING_DELTA = "model_reasoning_delta"
    MODEL_COMPLETED = "model_completed"
    RUN_BUDGET_WARNING = "run_budget_warning"
    RUN_BUDGET_FINALIZING = "run_budget_finalizing"
    RUN_BUDGET_EXCEEDED = "run_budget_exceeded"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_COMPLETED = "tool_approval_completed"
    MEMORY_REFLECTION_STARTED = "memory_reflection_started"
    MEMORY_REFLECTION_COMPLETED = "memory_reflection_completed"
    MEMORY_REFLECTION_FAILED = "memory_reflection_failed"
    MEMORY_REFLECTION_SKIPPED = "memory_reflection_skipped"
    MEMORY_ARCHIVE_STARTED = "memory_archive_started"
    MEMORY_ARCHIVE_COMPLETED = "memory_archive_completed"
    MEMORY_ARCHIVE_FAILED = "memory_archive_failed"
    MEMORY_ARCHIVE_SKIPPED = "memory_archive_skipped"
    SKILL_ACTIVATED = "skill_activated"
    SKILL_ACTIVATION_FAILED = "skill_activation_failed"
    CONTEXT_HANDOFF = "context_handoff"
    CONTEXT_COMPACTED = "context_compacted"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_CANCELLED = "agent_cancelled"


class AgentEvent(BaseModel):
    """一次 Agent 运行中的可序列化事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    conversation_id: str | None = None
    sequence: int = Field(default=0, ge=0)
    type: AgentEventType
    event_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    delta: str | None = None
    reasoning_delta: str | None = None
    message: Message | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    usage: ModelUsage | None = None
    stop_reason: AgentStopReason | None = None
    error: AgentError | None = None
    result: AgentResult | None = None
    approval_decision: ApprovalDecision | None = None
    rule_id: str | None = None
    rule_description: str | None = None
    original_estimated_input_tokens: int | None = Field(default=None, ge=0)
    prepared_input_tokens: int | None = Field(default=None, ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    context_trimmed: bool | None = None
    context_window: int | None = Field(default=None, ge=0)
    input_budget: int | None = Field(default=None, ge=0)
    working_input_budget: int | None = Field(default=None, ge=0)
    hard_trigger_tokens: int | None = Field(default=None, ge=0)
    hard_target_tokens: int | None = Field(default=None, ge=0)
    usage_ratio: float | None = Field(default=None, ge=0.0)
    trigger_tokens: int | None = Field(default=None, ge=0)
    target_tokens: int | None = Field(default=None, ge=0)
    tool_result_budget_tokens: int | None = Field(default=None, ge=0)
    tool_result_tokens_before: int | None = Field(default=None, ge=0)
    tool_result_tokens_after: int | None = Field(default=None, ge=0)
    tool_schema_tokens: int | None = Field(default=None, ge=0)
    message_tokens_before: int | None = Field(default=None, ge=0)
    message_tokens_after: int | None = Field(default=None, ge=0)
    unsummarized_conversation_blocks: int | None = Field(default=None, ge=0)
    conversation_block_limit: int | None = Field(default=None, gt=0)
    conversation_block_triggered: bool | None = None
    requires_compaction: bool | None = None
    exceeds_input_budget: bool | None = None
    capability_source: str | None = None
    original_usage_ratio: float | None = Field(default=None, ge=0.0)
    prepared_usage_ratio: float | None = Field(default=None, ge=0.0)
    compaction_stage: str | None = None
    compacted_tool_results: int | None = Field(default=None, ge=0)
    removed_tool_rounds: int | None = Field(default=None, ge=0)
    reached_target: bool | None = None
    needs_next_compaction_stage: bool | None = None
    summary_updated: bool | None = None
    summarized_conversation_blocks: int | None = Field(default=None, ge=0)
    summary_usage: ModelUsage | None = None
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = Field(default=None, ge=0.0)
    summary_error: str | None = None
    cache_prefix_reused: bool | None = None
    cache_prefix_message_count: int | None = Field(default=None, ge=0)
    reflection_triggered: bool | None = None
    reflection_action: str | None = None
    reflection_duration_ms: float | None = Field(default=None, ge=0.0)
    reflection_attempts: int | None = Field(default=None, ge=0)
    reflection_finish_reason: str | None = None
    reflection_error: str | None = None
    reflection_skip_reason: str | None = None
    reflection_memory_id: str | None = None
    reflection_mutation_applied: bool | None = None
    reflection_archive_required: bool | None = None
    reflection_retention_candidate_ids: tuple[str, ...] = ()
    reflection_input_json: str | None = None
    reflection_raw_output: str | None = None
    archive_triggered: bool | None = None
    archive_action: str | None = None
    archive_duration_ms: float | None = Field(default=None, ge=0.0)
    archive_error: str | None = None
    archive_skip_reason: str | None = None
    archive_memory_id: str | None = None
    archive_reason: str | None = None
    archive_active_count: int | None = Field(default=None, ge=0)
    archive_max_active: int | None = Field(default=None, gt=0)
    archive_candidate_ids: tuple[str, ...] = ()
    archive_remaining_overflow: int | None = Field(default=None, ge=0)
    skill_name: str | None = None
    skill_scope: str | None = None
    skill_error: str | None = None
    available_skill_count: int | None = Field(default=None, ge=0)
    skill_catalog_tokens: int | None = Field(default=None, ge=0)
    active_skill_names: tuple[str, ...] = ()
    active_skill_tokens: int | None = Field(default=None, ge=0)
    active_skill_message_names: tuple[str, ...] = ()
    # Memory 自动召回观测字段（每 Run 一次检索的候选与实际融合模式）。
    recall_candidate_ids: tuple[str, ...] = ()
    recall_mode: str | None = None
    run_budget_status: str | None = None
    run_budget_reason: str | None = None
    run_budget_chargeable_tokens: int | None = Field(default=None, ge=0)
    run_budget_model_calls: int | None = Field(default=None, ge=0)
    run_budget_warning_tokens: int | None = Field(default=None, ge=1)
    run_budget_finalization_tokens: int | None = Field(default=None, ge=1)
    run_budget_hard_tokens: int | None = Field(default=None, ge=1)
    run_budget_warning_model_calls: int | None = Field(default=None, ge=1)
    run_budget_finalization_model_calls: int | None = Field(default=None, ge=1)
    run_budget_hard_model_calls: int | None = Field(default=None, ge=1)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """运行 ID 必须是非空字符串。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("run_id cannot be empty")
        return normalized

    @field_validator("event_time")
    @classmethod
    def normalize_event_time(cls, value: datetime) -> datetime:
        """要求事件时间包含时区，并统一转换为 UTC。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include timezone information")
        return value.astimezone(UTC)


class AgentEventHandler(ABC):
    """接收 Agent 运行事件的异步处理器。"""

    @abstractmethod
    async def handle(self, event: AgentEvent) -> None:
        """接收一个已经完成编号的事件。"""


class NullEventHandler(AgentEventHandler):
    """忽略全部事件的默认处理器。"""

    async def handle(self, event: AgentEvent) -> None:
        pass


# class InMemoryEventHandler(AgentEventHandler):
#     """在内存中按发射顺序保存事件，主要用于测试和调试。"""

#     def __init__(self) -> None:
#         self._events: list[AgentEvent] = []

#     async def emit(self, event: AgentEvent) -> None:
#         self._events.append(event)

#     @property
#     def events(self) -> tuple[AgentEvent, ...]:
#         return tuple(self._events)

#     def clear(self) -> None:
#         self._events.clear()


class MultiEventHandlers(AgentEventHandler):
    """把同一个事件依次发送给多个相互隔离的处理器。"""

    def __init__(self, *handlers: AgentEventHandler) -> None:
        self._handlers = handlers

    async def handle(self, event: AgentEvent) -> None:
        for handler in self._handlers:
            try:
                await handler.handle(event)
            except Exception:
                continue


__all__ = [
    "AgentEvent",
    "AgentEventHandler",
    "AgentEventType",
    "MultiEventHandlers",
    # "InMemoryEventHandler",
    # "NullEventHandler",
]
