"""从 durable AgentEvent 构建 Run 级 Usage 账本。"""

from __future__ import annotations

from collections.abc import Sequence

from app.agent.budget import chargeable_tokens
from app.agent.events import AgentEvent, AgentEventType
from app.agent.utils import add_model_usage
from app.model.config import ModelUsage

from .models import RunUsageSummary

_REFLECTION_USAGE_EVENTS = frozenset(
    {
        AgentEventType.MEMORY_REFLECTION_COMPLETED,
        AgentEventType.MEMORY_REFLECTION_FAILED,
    }
)
_MAINTENANCE_USAGE_EVENTS = frozenset(
    {
        AgentEventType.MEMORY_ARCHIVE_COMPLETED,
        AgentEventType.MEMORY_ARCHIVE_FAILED,
    }
)


def summarize_run_usage(events: Sequence[AgentEvent]) -> RunUsageSummary:
    """按 Main/Post-Run 边界聚合事件中的 Provider Usage。"""

    main_agent = _main_agent_usage(events)
    context_summary = _context_summary_usage(events)
    reflection = _sum_event_usage(events, _REFLECTION_USAGE_EVENTS)
    maintenance = _sum_event_usage(events, _MAINTENANCE_USAGE_EVENTS)
    reflection_status, reflection_skip_reason = _reflection_status(events)
    budget = _run_budget_snapshot(events)
    provider_total = add_model_usage(
        add_model_usage(
            add_model_usage(main_agent, context_summary),
            reflection,
        ),
        maintenance,
    )
    summary_status, summary_provider, summary_model, summary_duration_ms = (
        _context_summary_status(events)
    )
    return RunUsageSummary(
        main_agent=main_agent,
        context_summary=context_summary,
        memory_reflection=reflection,
        memory_maintenance=maintenance,
        provider_total=provider_total,
        tool_schema_tokens_estimated=sum(
            event.tool_schema_tokens or 0
            for event in events
            if event.type is AgentEventType.MODEL_STARTED
        ),
        memory_reflection_status=reflection_status,
        memory_reflection_skip_reason=reflection_skip_reason,
        context_summary_status=summary_status,
        context_summary_provider=summary_provider,
        context_summary_model=summary_model,
        context_summary_duration_ms=summary_duration_ms,
        main_agent_chargeable_tokens=_main_agent_chargeable_tokens(
            events,
            fallback=main_agent,
        ),
        **budget,
    )


def _run_budget_snapshot(events: Sequence[AgentEvent]) -> dict[str, object]:
    """读取最后一个带预算字段的事件，保留当时 Runtime 的策略快照。"""

    latest: AgentEvent | None = None
    for event in events:
        if event.run_budget_status is not None:
            latest = event
    if latest is None:
        return {}
    return {
        "run_budget_status": latest.run_budget_status,
        "run_budget_reason": latest.run_budget_reason,
        "run_budget_warning_tokens": latest.run_budget_warning_tokens,
        "run_budget_finalization_tokens": (
            latest.run_budget_finalization_tokens
        ),
        "run_budget_hard_tokens": latest.run_budget_hard_tokens,
        "run_budget_warning_model_calls": (
            latest.run_budget_warning_model_calls
        ),
        "run_budget_finalization_model_calls": (
            latest.run_budget_finalization_model_calls
        ),
        "run_budget_hard_model_calls": latest.run_budget_hard_model_calls,
    }


def _main_agent_chargeable_tokens(
    events: Sequence[AgentEvent],
    *,
    fallback: ModelUsage,
) -> int:
    """逐次计算预算Token，避免一次未知缓存破坏其它调用的已知细分。"""

    values: list[ModelUsage] = []
    for event in events:
        if event.type is AgentEventType.MODEL_COMPLETED and event.usage is not None:
            values.append(event.usage)
        if (
            event.type is AgentEventType.MODEL_STARTED
            and event.summary_usage is not None
            and _has_tokens(event.summary_usage)
        ):
            values.append(event.summary_usage)
    if not values:
        return chargeable_tokens(fallback)
    return sum(chargeable_tokens(usage) for usage in values)


def _reflection_status(events: Sequence[AgentEvent]) -> tuple[str, str | None]:
    status = "not_run"
    skip_reason: str | None = None
    for event in events:
        if event.type is AgentEventType.MEMORY_REFLECTION_STARTED:
            status = "running"
        elif event.type is AgentEventType.MEMORY_REFLECTION_COMPLETED:
            status = "completed"
            skip_reason = None
        elif event.type is AgentEventType.MEMORY_REFLECTION_FAILED:
            status = "failed"
            skip_reason = None
        elif event.type is AgentEventType.MEMORY_REFLECTION_SKIPPED:
            status = "skipped"
            skip_reason = event.reflection_skip_reason
    return status, skip_reason


def _main_agent_usage(events: Sequence[AgentEvent]) -> ModelUsage:
    usage = ModelUsage()
    completed_calls = 0
    for event in events:
        if event.type is AgentEventType.MODEL_COMPLETED and event.usage is not None:
            usage = add_model_usage(usage, _with_inferred_call(event.usage))
            completed_calls += 1
    if completed_calls:
        return usage

    terminal_usage: ModelUsage | None = None
    for event in events:
        if event.type not in {
            AgentEventType.AGENT_COMPLETED,
            AgentEventType.AGENT_FAILED,
            AgentEventType.AGENT_CANCELLED,
        }:
            continue
        terminal_usage = event.result.usage if event.result is not None else event.usage

    if terminal_usage is not None:
        if terminal_usage.model_calls > 0:
            return terminal_usage
        # 旧 Trace 没有 model_calls；用持久化完成事件回算，保持历史可分析。
        return terminal_usage.model_copy(
            update={"model_calls": _main_model_call_count(events)}
        )

    return ModelUsage()


def _main_model_call_count(events: Sequence[AgentEvent]) -> int:
    return sum(
        event.usage.model_calls or 1
        for event in events
        if event.type is AgentEventType.MODEL_COMPLETED and event.usage is not None
    )


def _context_summary_usage(events: Sequence[AgentEvent]) -> ModelUsage:
    usage = ModelUsage()
    for event in events:
        if (
            event.type is AgentEventType.MODEL_STARTED
            and event.summary_usage is not None
        ):
            usage = add_model_usage(usage, _with_inferred_call(event.summary_usage))
    return usage


def _context_summary_status(
    events: Sequence[AgentEvent],
) -> tuple[str, str | None, str | None, float]:
    status = "not_run"
    provider: str | None = None
    model: str | None = None
    duration_ms = 0.0
    for event in events:
        if event.type is not AgentEventType.MODEL_STARTED:
            continue
        if event.summary_provider is not None:
            provider = event.summary_provider
        if event.summary_model is not None:
            model = event.summary_model
        if event.summary_duration_ms is not None:
            duration_ms += event.summary_duration_ms
        if event.summary_updated:
            status = "completed"
        elif event.summary_error is not None:
            status = "failed"
        elif event.summary_usage is not None and _has_tokens(event.summary_usage):
            status = "completed"
    return status, provider, model, duration_ms


def _sum_event_usage(
    events: Sequence[AgentEvent],
    event_types: frozenset[AgentEventType],
) -> ModelUsage:
    usage = ModelUsage()
    for event in events:
        if event.type in event_types and event.usage is not None:
            usage = add_model_usage(usage, _with_inferred_call(event.usage))
    return usage


def _with_inferred_call(usage: ModelUsage) -> ModelUsage:
    if usage.model_calls > 0 or not _has_tokens(usage):
        return usage
    return usage.model_copy(update={"model_calls": 1})


def _has_tokens(usage: ModelUsage) -> bool:
    return bool(usage.input_tokens or usage.output_tokens or usage.total_tokens)


__all__ = ["summarize_run_usage"]
