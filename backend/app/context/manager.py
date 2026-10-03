
'''
读取模型容量，计算输入预算与压缩目标
    ↓
正常历史路径：用已有摘要替换已覆盖的旧历史
    ↓
估算完整请求、消息、工具 Schema、工具结果的 token
    ↓
工具结果超预算？
    └─ ToolReducer
         先截短旧工具结果
         仍超预算则整轮移除较早工具调用
         保留近期受保护轮次
    ↓
历史对话需要压缩？
    └─ ConversationReducer
         找到本次可以摘要的历史范围
         旧摘要 + 新增历史 → 新摘要
         用新摘要替换旧历史前缀
    ↓
重新统计最终用量，检查输入硬上限
    ↓
返回 ContextDecision
'''

from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum

from collections.abc import Sequence

from app.model.config import Message
from app.context.blocks import ConversationBlock, partition_messages
from app.context.budget import ContextBudgetPolicy, build_budget_policy
from app.context.capability import (
    ModelCapabilitiesRegistry,
    build_model_capability_registry,
)
from app.context.config import ContextSettings
from app.context.reducers.conversation import (
    ConversationReducer,
    build_summary_candidate,
)
from app.context.reducers.tools import ToolReducer
from app.context.summary import ConversationSummaryState
from app.context.tokens import TokenEstimator, default_token_estimator
from app.model.config import ModelUsage
from app.tools.config import ToolDefinition
from .task_boundary import TaskBoundaryTrigger

# Public name for callers that select the context-compaction policy explicitly.
ContextCompactionTrigger = TaskBoundaryTrigger


class ContextCompactionStage(StrEnum):
    """本次模型请求实际执行到的压缩阶段。"""

    NONE = "none"
    TOOL_RESULTS = "tool_results"
    TOOL_ROUNDS = "tool_rounds"
    TOOL_RESULTS_AND_ROUNDS = "tool_results_and_rounds"
    ROLLING_SUMMARY = "rolling_summary"
    TOOL_AND_ROLLING_SUMMARY = "tool_and_rolling_summary"


@dataclass(frozen=True)
class ContextDecision:
    """一次模型调用最终发送的上下文与预算状态。"""

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    provider: str | None = None
    model: str | None = None
    original_estimated_input_tokens: int | None = None
    prepared_input_tokens: int | None = None
    estimated_input_tokens: int | None = None
    context_window: int | None = None
    reserved_output_tokens: int | None = None
    safety_margin_tokens: int | None = None
    input_budget: int | None = None
    working_input_budget: int | None = None
    hard_trigger_tokens: int | None = None
    hard_target_tokens: int | None = None
    trigger_tokens: int | None = None
    target_tokens: int | None = None
    tool_result_budget_tokens: int | None = None
    tool_result_tokens_before: int = 0
    tool_result_tokens_after: int = 0
    tool_schema_tokens: int = 0
    message_tokens_before: int = 0
    message_tokens_after: int = 0
    unsummarized_conversation_blocks: int = 0
    conversation_block_limit: int | None = None
    conversation_block_triggered: bool = False
    original_usage_ratio: float | None = None
    prepared_usage_ratio: float | None = None
    usage_ratio: float | None = None
    requires_compaction: bool = False
    exceeds_input_budget: bool = False
    capability_source: str | None = None
    trimmed: bool = False
    compaction_stage: ContextCompactionStage = ContextCompactionStage.NONE
    reached_target: bool = True
    needs_next_compaction_stage: bool = False
    needs_handoff: bool = False
    compacted_tool_results: int = 0
    removed_tool_rounds: int = 0
    summary_state: ConversationSummaryState | None = None
    summary_updated: bool = False
    summarized_conversation_blocks: int = 0
    summary_usage: ModelUsage = field(default_factory=ModelUsage)
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = None
    summary_error: str | None = None
    reason: str | None = None


class ContextManager:
    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        *,
        registry: ModelCapabilitiesRegistry | None = None,
        budget_policy: ContextBudgetPolicy | None = None,
        context_settings: ContextSettings | None = None,
        keep_recent_tool_rounds: int | None = None,
        tool_reducer: ToolReducer | None = None,
        conversation_reducer: ConversationReducer | None = None,
    ) -> None:
        settings = context_settings or ContextSettings()
        keep_recent_tool_rounds = (
            settings.context_keep_recent_tool_rounds
            if keep_recent_tool_rounds is None
            else keep_recent_tool_rounds
        )
        if keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        self._estimator = estimator or default_token_estimator()
        self._registry = registry or build_model_capability_registry(
            context_settings=settings
        )
        self._budget_policy = budget_policy or build_budget_policy(settings)
        self._tool_reducer = tool_reducer or ToolReducer( # 截断过长输出、缩减较早的工具结果
            keep_recent_tool_rounds=keep_recent_tool_rounds,
            max_tool_result_chars=settings.context_max_tool_result_chars,
            tool_result_head_chars=settings.context_tool_result_head_chars,
            tool_result_tail_chars=settings.context_tool_result_tail_chars,
        )
        self._conversation_reducer = conversation_reducer # 把较早的对话压缩为摘要，保留近期对话
        self._max_unsummarized_conversation_blocks = (
            settings.context_max_unsummarized_conversation_blocks
        )
        self.task_boundary_required_stable_count = settings.context_task_boundary_stable_count

    def context_window_for(self, provider: str, model: str) -> int:
        """Return the configured context window for a provider/model pair."""
        return self._registry.lookup(provider, model).context_window
    
    async def prepare(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
        max_output_tokens: int | None = None,
        history_count: int | None = None,
        keep_recent_tool_rounds: int | None = None,
        summary_state: ConversationSummaryState | None = None,
        handoff: bool = False,
        trigger: TaskBoundaryTrigger = TaskBoundaryTrigger.AUTO,
    ) -> ContextDecision:
        trigger = TaskBoundaryTrigger(trigger)

        if history_count is None:
            history_count = 0
        if history_count < 0 or history_count > len(messages):
            raise ValueError("history_count must be within the messages range")
        keep_recent_tool_rounds = (
            self._tool_reducer.keep_recent_tool_rounds
            if keep_recent_tool_rounds is None
            else keep_recent_tool_rounds
        )
        if keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")

        raw_messages = tuple(messages)
        raw_history = raw_messages[:history_count]
        current_messages = raw_messages[history_count:]
        valid_summary_state = (
            summary_state
            if summary_state is None
            or summary_state.covered_message_count <= len(raw_history)
            else None
        )

        original_messages, _ = build_summary_candidate(
            raw_history,
            current_messages,
            valid_summary_state,
        )

        capabilities = self._registry.lookup(provider, model)
        budget = self._budget_policy.compute(
            capabilities,
            max_output_tokens=max_output_tokens,
        )
        original_estimated = self._estimator.estimate_request(
            original_messages,
            tools=tools,
            model=model,
            provider=provider,
        )
        original_usage_ratio = (
            original_estimated / budget.input_budget
            if budget.input_budget > 0
            else None
        )
        tool_schema_tokens = self._estimator.estimate_tools(tools, model = model, provider = provider)
        message_tokens = self._estimator.estimate_messages(original_messages, model = model, provider = provider)
        tool_result_tokens = self._estimator.estimate_tool_results(
            original_messages,
            model=model,
            provider=provider,
        )

        request_messages = original_messages
        prepared_input_tokens = original_estimated
        compacted_tool_results = 0
        removed_tool_rounds = 0
        tool_result_tokens_before = tool_result_tokens
        tool_result_tokens_after = tool_result_tokens
        tool_results_requires_reduction = (
            tool_result_tokens_before > budget.tool_result_budget_tokens
        )
        task_triggered = trigger is TaskBoundaryTrigger.TASK

        if tool_results_requires_reduction or task_triggered:
            tool_results_reduction = self._tool_reducer.project(
                original_messages,
                original_estimated=prepared_input_tokens,
                tool_result_tokens=tool_result_tokens,
                tool_result_budget_tokens=budget.tool_result_budget_tokens,
                tools=tools,
                model=model,
                provider=provider,
                keep_recent_tool_rounds=keep_recent_tool_rounds,
                force=task_triggered,
            )
            request_messages = tool_results_reduction.messages
            prepared_input_tokens = tool_results_reduction.estimated_input_tokens
            compacted_tool_results = tool_results_reduction.compacted_tool_results
            removed_tool_rounds = tool_results_reduction.removed_tool_rounds
            tool_result_tokens_after = tool_results_reduction.tool_result_tokens_after


        covered_message_count = (
            valid_summary_state.covered_message_count
            if valid_summary_state is not None
            else 0
        )
        unsummarized_conversation_blocks = sum(
            isinstance(block, ConversationBlock)
            for block in partition_messages(raw_history[covered_message_count:])
        )
        conversation_block_triggered = (
            unsummarized_conversation_blocks
            > self._max_unsummarized_conversation_blocks
        )
        conversation_requires_compaction = (
            task_triggered
            or prepared_input_tokens >= budget.trigger_tokens
            or conversation_block_triggered
        )

        requires_compaction = (
            tool_results_requires_reduction or conversation_requires_compaction
        )
        current_summary_state = valid_summary_state
        summary_updated = False
        summarized_conversation_blocks = 0
        summary_usage = ModelUsage()
        summary_error: str | None = None
        summary_provider: str | None = None
        summary_model: str | None = None
        summary_duration_ms: float | None = None
        compaction_stage = ContextCompactionStage.NONE
        reached_target = not requires_compaction or (
            prepared_input_tokens <= budget.target_tokens
            and tool_result_tokens_after <= budget.tool_result_budget_tokens
        )
        if compacted_tool_results and removed_tool_rounds:
            compaction_stage = ContextCompactionStage.TOOL_RESULTS_AND_ROUNDS
        elif removed_tool_rounds:
            compaction_stage = ContextCompactionStage.TOOL_ROUNDS
        elif compacted_tool_results:
            compaction_stage = ContextCompactionStage.TOOL_RESULTS

        if (conversation_requires_compaction or handoff) and self._conversation_reducer is not None:
            conversation_reduction = await self._conversation_reducer.reduce(
                raw_history=raw_history,
                prepared_messages=request_messages,
                current_messages=current_messages,
                previous_state=valid_summary_state,
                # TASK 是一次旧任务切换压缩：复用当前 summary 字段和 reducer，
                # 但不保留旧的近期对话块；最新用户消息仍在 current_messages 中。
                handoff=handoff or task_triggered,
                force=task_triggered,
                initial_estimated_input_tokens=prepared_input_tokens,
                target_tokens=budget.target_tokens,
                tools=tuple(tools),
                model=model,
                provider=provider,
            )
            request_messages = conversation_reduction.messages
            prepared_input_tokens = conversation_reduction.estimated_input_tokens
            current_summary_state = conversation_reduction.summary_state
            summarized_conversation_blocks = (
                conversation_reduction.summarized_conversation_blocks
            )
            summary_updated = current_summary_state != valid_summary_state
            summary_usage = conversation_reduction.summary_usage
            summary_error = conversation_reduction.error
            summary_provider = conversation_reduction.summary_provider
            summary_model = conversation_reduction.summary_model
            summary_duration_ms = conversation_reduction.summary_duration_ms
            reached_target = (
                conversation_reduction.reached_target
                and tool_result_tokens_after <= budget.tool_result_budget_tokens
            )
            if summary_updated:
                if compaction_stage is ContextCompactionStage.NONE:
                    compaction_stage = ContextCompactionStage.ROLLING_SUMMARY
                else:
                    compaction_stage = ContextCompactionStage.TOOL_AND_ROLLING_SUMMARY
            
        prepared_usage_ratio = (
            prepared_input_tokens / budget.input_budget
            if budget.input_budget > 0
            else None
        )
        trimmed = request_messages != raw_messages
        exceeds_input_budget = prepared_input_tokens > budget.input_budget
        message_tokens_after = self._estimator.estimate_messages(
            request_messages,
            model=model,
            provider=provider,
        )
        tool_result_tokens_after = self._estimator.estimate_tool_results(
            request_messages,
            model=model,
            provider=provider,
        )
        reached_target = not requires_compaction or (
            prepared_input_tokens <= budget.target_tokens
            and tool_result_tokens_after <= budget.tool_result_budget_tokens
        )
        needs_next_compaction_stage = requires_compaction and not reached_target
        needs_handoff = (
            exceeds_input_budget and not handoff
            and self._conversation_reducer is not None
        )

        reason = (
            "[预算] "
            f"input_budget={budget.input_budget}; "
            f"working_input_budget={budget.working_input_budget}; "
            f"trigger={budget.trigger_tokens}; "
            f"target={budget.target_tokens}; "
            f"tool_result_budget={budget.tool_result_budget_tokens}"

            " | [处理前] "
            f"original_estimated={original_estimated}; "
            f"tool_result_tokens_before={tool_result_tokens_before}; "
            f"unsummarized_conversation_blocks={unsummarized_conversation_blocks}"

            " | [触发条件] "
            f"tool_requires_reduction={tool_results_requires_reduction}; "
            f"conversation_requires_compaction={conversation_requires_compaction}; "
            f"conversation_block_triggered={conversation_block_triggered}; "
            f"requires_compaction={requires_compaction}"

            " | [处理动作] "
            f"compaction_stage={compaction_stage.value}; "
            f"trimmed={trimmed}; "
            f"compacted_tool_results={compacted_tool_results}; "
            f"removed_tool_rounds={removed_tool_rounds}; "
            f"summary_updated={summary_updated}; "
            f"summarized_conversation_blocks={summarized_conversation_blocks}"

            " | [处理后] "
            f"prepared_input_tokens={prepared_input_tokens}; "
            f"tool_result_tokens_after={tool_result_tokens_after}; "
            f"exceeds_input_budget={exceeds_input_budget}; "
            f"reached_target={reached_target}; "
            f"needs_next_compaction_stage={needs_next_compaction_stage}; "
            f"needs_handoff={needs_handoff}"
        )
        return ContextDecision(
            messages=request_messages,
            tools=tuple(tools),
            provider=capabilities.provider,
            model=capabilities.model,
            original_estimated_input_tokens=original_estimated,
            prepared_input_tokens=prepared_input_tokens,
            estimated_input_tokens=prepared_input_tokens,
            context_window=budget.context_window,
            reserved_output_tokens=budget.reserved_output_tokens,
            safety_margin_tokens=budget.safety_margin_tokens,
            input_budget=budget.input_budget,
            working_input_budget=budget.working_input_budget,
            hard_trigger_tokens=budget.hard_trigger_tokens,
            hard_target_tokens=budget.hard_target_tokens,
            trigger_tokens=budget.trigger_tokens,
            target_tokens=budget.target_tokens,
            tool_result_budget_tokens=budget.tool_result_budget_tokens,
            tool_result_tokens_before=tool_result_tokens_before,
            tool_result_tokens_after=tool_result_tokens_after,
            tool_schema_tokens=tool_schema_tokens,
            message_tokens_before=message_tokens,
            message_tokens_after=message_tokens_after,
            unsummarized_conversation_blocks=(
                unsummarized_conversation_blocks
            ),
            conversation_block_limit=(
                self._max_unsummarized_conversation_blocks
            ),
            conversation_block_triggered=conversation_block_triggered,
            original_usage_ratio=original_usage_ratio,
            prepared_usage_ratio=prepared_usage_ratio,
            usage_ratio=prepared_usage_ratio,
            requires_compaction=requires_compaction,
            exceeds_input_budget=exceeds_input_budget,
            capability_source=capabilities.source.value,
            trimmed=trimmed,
            compaction_stage=compaction_stage,
            reached_target=reached_target,
            needs_next_compaction_stage=needs_next_compaction_stage,
            needs_handoff=needs_handoff,
            compacted_tool_results=compacted_tool_results,
            removed_tool_rounds=removed_tool_rounds,
            summary_state=current_summary_state,
            summary_updated=summary_updated,
            summarized_conversation_blocks=summarized_conversation_blocks,
            summary_usage=summary_usage,
            summary_provider=summary_provider,
            summary_model=summary_model,
            summary_duration_ms=summary_duration_ms,
            summary_error=summary_error,
            reason=reason,
        )
