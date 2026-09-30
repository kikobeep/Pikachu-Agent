"""基于滚动摘要的普通对话第二层压缩器。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.model.config import Message, MessageRole
from app.memory.utils import add_usage
from app.model.config import ModelUsage
from app.tools.config import ToolDefinition

from ..blocks import (
    ConversationBlock,
    OrphanToolBlock,
    MessageBlock,
    ToolRoundBlock,
    partition_messages,
)
from ..summarize import ContextSummarizer
from ..summary import (
    SUMMARY_MESSAGE_NAME,
    ConversationSummaryState,
)
from ..tokens import TokenEstimator, default_token_estimator


@dataclass(frozen=True)
class ConversationReductionResult:
    """第二层滚动摘要的请求上下文、状态和统计。"""

    messages: tuple[Message, ...]
    estimated_input_tokens: int
    summary_state: ConversationSummaryState | None = None
    summarized_conversation_blocks: int = 0
    summary_usage: ModelUsage = field(default_factory=ModelUsage)
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_duration_ms: float | None = None
    reached_target: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _IndexedBlock:
    block: MessageBlock
    start: int
    end: int


class ConversationReducer:
    def __init__(
        self,
        summarizer: ContextSummarizer,
        *,
        estimator: TokenEstimator | None = None,
        keep_recent_conversation_blocks: int = 4,
        keep_recent_tool_rounds: int = 2,
    ) -> None:
        if keep_recent_conversation_blocks < 0:
            raise ValueError("keep_recent_conversation_blocks cannot be negative")
        if keep_recent_tool_rounds < 0:
            raise ValueError("keep_recent_tool_rounds cannot be negative")
        self._estimator = estimator if estimator is not None else default_token_estimator()
        self._summarizer = summarizer
        self.summary_provider = summarizer.provider_hint
        
        self.summary_model = summarizer.model_hint
        self.keep_recent_conversation_blocks = keep_recent_conversation_blocks
        self.keep_recent_tool_rounds = keep_recent_tool_rounds

    async def reduce(
        self,
        *,
        raw_history: Sequence[Message],
        prepared_messages: Sequence[Message],
        current_messages: Sequence[Message],
        previous_state: ConversationSummaryState | None,
        initial_estimated_input_tokens: int,
        target_tokens: int,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
        handoff: bool = False,
    ) -> ConversationReductionResult:
        """摘要可覆盖的最旧前缀；失败时原样返回 prepared_messages。

        初始用量与内部重算须使用相同的估算器及 tools/model/provider。
        model/provider 是待压缩的主模型请求配置，而不是摘要模型配置。
        """

        raw = tuple(raw_history)
        prepared = tuple(prepared_messages)
        covered = previous_state.covered_message_count if previous_state else 0
        if covered > len(raw):
            return ConversationReductionResult(
                    messages=prepared,
                    estimated_input_tokens=initial_estimated_input_tokens,
                    summary_state=previous_state,
                    reached_target=initial_estimated_input_tokens <= target_tokens,
                    error="summary covered_message_count exceeds current history length",
                )

        indexed = _index_blocks(raw[covered:], offset=covered) # 调用 partition_messages() 分块，再给每个块记录 start、end
        cutoff = _summary_cutoff( # 根据保留规则，找出最早需要保护的消息块的起点
            indexed,
            keep_recent_conversation_blocks=0 if handoff else self.keep_recent_conversation_blocks,
            keep_recent_tool_rounds=0 if handoff else self.keep_recent_tool_rounds,
            history_length=len(raw),
        )
        summary_blocks = [
            item
            for item in indexed
            if item.end <= cutoff and isinstance(item.block, (ConversationBlock, ToolRoundBlock))
        ]
        if cutoff <= covered or not summary_blocks:
            return ConversationReductionResult(
                    messages=prepared,
                    estimated_input_tokens=initial_estimated_input_tokens,
                    summary_state=previous_state,
                    reached_target=initial_estimated_input_tokens <= target_tokens,
                )
          

        source_messages = tuple(
            message for item in summary_blocks for message in item.block.messages
        )
        previous_summary = previous_state.summary if previous_state else None
        total_usage = ModelUsage()
        last_error = "summary generation failed"
        started = time.perf_counter()
        for attempt in range(2):
            try:
                generated = (
                    await self._summarizer.summarize(
                        previous_summary,
                        source_messages,
                    )
                    if attempt == 0
                    else await self._summarizer.retry_compact(
                        previous_summary,
                        source_messages,
                        reason=last_error,
                    )
                )
            except Exception as exc:
                total_usage = _add_usage(total_usage, _error_usage(exc))
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            total_usage = _add_usage(total_usage, generated.usage)
            new_state = ConversationSummaryState(
                summary=generated.summary,
                covered_message_count=cutoff,
            )
            reduced_messages = _replace_covered_prefix(
                prepared,
                raw[covered:cutoff],
                new_state,
            )
            estimated = self._estimator.estimate_request(
                reduced_messages, tools=tools, model=model, provider=provider,
            )
            if estimated < initial_estimated_input_tokens:
                return ConversationReductionResult(
                    messages=reduced_messages,
                    estimated_input_tokens=estimated,
                    summary_state=new_state,
                    summarized_conversation_blocks=sum(
                        isinstance(item.block, ConversationBlock) for item in summary_blocks
                    ),
                    summary_usage=total_usage,
                    summary_provider=self.summary_provider,
                    summary_model=self.summary_model,
                    summary_duration_ms=(time.perf_counter() - started) * 1000,
                    reached_target=estimated <= target_tokens,
                )
            last_error = "generated summary did not reduce the request context"

        return self._unchanged(
            prepared,
            previous_state,
            initial_estimated_input_tokens,
            target_tokens,
            last_error,
            summary_usage=total_usage,
            summary_provider=self.summary_provider,
            summary_model=self.summary_model,
            summary_duration_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _unchanged(
        messages: tuple[Message, ...],
        state: ConversationSummaryState | None,
        estimated: int,
        target_tokens: int,
        error: str | None = None,
        *,
        summary_usage: ModelUsage | None = None,
        summary_provider: str | None = None,
        summary_model: str | None = None,
        summary_duration_ms: float | None = None,
    ) -> ConversationReductionResult:
        return ConversationReductionResult(
            messages=messages,
            estimated_input_tokens=estimated,
            summary_state=state,
            summary_usage=summary_usage or ModelUsage(),
            summary_provider=summary_provider,
            summary_model=summary_model,
            summary_duration_ms=summary_duration_ms,
            reached_target=estimated <= target_tokens,
            error=error,
        )


def _error_usage(error: Exception) -> ModelUsage:
    usage = getattr(error, "usage", None)
    return usage if isinstance(usage, ModelUsage) else ModelUsage()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    return add_usage(left, right)


def build_summary_candidate(
    raw_history: Sequence[Message],
    current_messages: Sequence[Message],
    state: ConversationSummaryState | None,
) -> tuple[tuple[Message, ...], int]:
    """用已持久化摘要替换其覆盖的原始历史前缀。"""

    history = tuple(raw_history)
    current = tuple(current_messages)
    if state is None or state.covered_message_count == 0:
        return (*history, *current), len(history)
    if state.covered_message_count > len(history):
        return (*history, *current), len(history)

    covered_prefix = history[: state.covered_message_count]
    preserved_system = tuple(
        message
        for message in covered_prefix
        if message.role is MessageRole.SYSTEM and message.name != SUMMARY_MESSAGE_NAME
    )
    prepared_history = (
        *preserved_system,
        state.summary.to_message(),
        *history[state.covered_message_count :],
    )
    return (*prepared_history, *current), len(prepared_history)


def _index_blocks(
    messages: Sequence[Message],
    *,
    offset: int,
) -> tuple[_IndexedBlock, ...]:
    indexed: list[_IndexedBlock] = []
    position = offset
    for block in partition_messages(messages):
        end = position + len(block.messages)
        indexed.append(_IndexedBlock(block=block, start=position, end=end))
        position = end
    return tuple(indexed)


def _summary_cutoff(
    blocks: Sequence[_IndexedBlock],
    *,
    keep_recent_conversation_blocks: int,
    keep_recent_tool_rounds: int,
    history_length: int,
) -> int:
    conversation_indices = [
        index
        for index, item in enumerate(blocks)
        if isinstance(item.block, ConversationBlock)
    ]
    tool_indices = [
        index
        for index, item in enumerate(blocks)
        if isinstance(item.block, ToolRoundBlock)
    ]
    protected: set[int] = {
        index
        for index, item in enumerate(blocks)
        if isinstance(item.block, OrphanToolBlock)
    }
    if keep_recent_conversation_blocks:
        protected.update(conversation_indices[-keep_recent_conversation_blocks:])
    if keep_recent_tool_rounds:
        recent_tool_indices = tool_indices
        if (
            keep_recent_conversation_blocks
            and len(conversation_indices) > keep_recent_conversation_blocks
        ):
            last_summarizable_conversation = conversation_indices[
                -keep_recent_conversation_blocks - 1
            ]
            recent_tool_indices = [
                index
                for index in tool_indices
                if index > last_summarizable_conversation
            ]
        protected.update(recent_tool_indices[-keep_recent_tool_rounds:])
    if not protected:
        return history_length
    return min(blocks[index].start for index in protected)


def _replace_covered_prefix(
    prepared_messages: Sequence[Message],
    covered_raw_messages: Sequence[Message],
    state: ConversationSummaryState,
) -> tuple[Message, ...]:
    removable_ids = {
        id(message)
        for message in covered_raw_messages
        if message.role is not MessageRole.SYSTEM
    }
    covered_call_ids = {
        call.id for message in covered_raw_messages for call in message.tool_calls
    }
    kept: list[Message] = []
    for message in prepared_messages:
        if message.name == SUMMARY_MESSAGE_NAME:
            continue
        if id(message) in removable_ids:
            continue
        if message.tool_call_id in covered_call_ids:
            continue
        if message.tool_calls and all(
            call.id in covered_call_ids for call in message.tool_calls
        ):
            continue
        kept.append(message)

    insertion_index = 0
    while (
        insertion_index < len(kept) and kept[insertion_index].role is MessageRole.SYSTEM
    ):
        insertion_index += 1
    kept.insert(insertion_index, state.summary.to_message())
    return tuple(kept)


__all__ = [
    "ConversationReducer",
    "ConversationReductionResult",
    "build_summary_candidate",
]
