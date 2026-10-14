"""FlowState 风格的轻量 Agent Loop。

它复用现有工具执行、权限、预算、checkpoint 和事件链路，但不把完整 history
传给模型；模型只看到状态索引、当前工作集和最新观察。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from app.agent.budget import RunBudget, chargeable_tokens
from app.agent.context_injection import ContextRuntime
from app.agent.emitter import EventEmitter
from app.agent.events import AgentEventType
from app.agent.error import (
    MaxStepsExceededError,
    ModelInvocationError,
    RepeatedToolCallError,
    RunBudgetExceededError,
)
from app.agent.result import AgentError, AgentResult, AgentStopReason, ToolCallRecord, ToolRound
from app.agent.tool_hooks import AgentEventHook
from app.agent.tools_round_executor import ToolRoundExecutor
from app.agent.utils import (
    add_usage,
    run_budget_detail,
    run_budget_event_fields,
    tool_call_signature,
)
from app.checkpoint.config import RunCheckpoint
from app.checkpoint.store import CheckpointStore
from app.context.summary import ConversationSummaryState
from app.model.config import AgentMode, Message, MessageRole, ModelRequest, ModelUsage, ToolCall
from app.model.registry import ModelAdapterRegistry
from app.tools.register import ToolRegistry, ensure_tool_search_registered

from .context import (
    normalize_content_tool_calls,
    parse_model_output,
    render_active_state,
    render_historical_index,
    render_observation,
    validate_state_delta,
)
from .models import LightToolCall
from .prompt import LIGHT_STATE_PROMPT
from .store import LightStateStore

logger = logging.getLogger(__name__)


class LightAgentLoop:
    """只在 ``AgentMode.LIGHT`` 使用的状态驱动循环。"""

    def __init__(
        self,
        *,
        model_registry: ModelAdapterRegistry,
        tool_registry: ToolRegistry,
        tool_executor,
        provider,
        model: str | None,
        system_prompt: str | None,
        max_steps: int,
        max_tool_rounds: int | None,
        max_output_tokens: int | None,
        checkpoint_store: CheckpointStore | None,
        run_budget: RunBudget,
        state_store: LightStateStore,
    ) -> None:
        self._model_registry = model_registry
        self._tool_registry = tool_registry
        self._tool_round_executor = ToolRoundExecutor(
            registry=tool_registry,
            executor=tool_executor,
            checkpoint_store=checkpoint_store,
        )
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._max_tool_rounds = max_tool_rounds
        self._max_output_tokens = max_output_tokens
        self._checkpoint_store = checkpoint_store
        self._run_budget = run_budget
        self._state_store = state_store

    def set_max_output_tokens(self, value: int | None) -> None:
        if value is not None and value < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self._max_output_tokens = value

    async def run(
        self,
        run_id: str,
        user_input: str,
        *,
        history: Sequence[Message],
        conversation_id: str | None,
        emitter: EventEmitter,
        summary_state: ConversationSummaryState | None,
        recovery_checkpoint: RunCheckpoint | None,
        mode: AgentMode,
        additional_system_prompt: str | None = None,
    ) -> AgentResult:
        del summary_state
        if mode is not AgentMode.LIGHT:
            raise ValueError("LightAgentLoop requires AgentMode.LIGHT")

        scope_id = conversation_id or run_id
        adapter = self._model_registry.get(self._provider)
        resolved_model = self._model or adapter.default_model
        resolved_provider = adapter.provider
        output_tokens = self._max_output_tokens or adapter.config.default_max_output_tokens
        effective_prompt = self._system_prompt or ""
        if additional_system_prompt and additional_system_prompt.strip():
            effective_prompt = (
                f"{effective_prompt}\n\n{additional_system_prompt.strip()}"
                if effective_prompt
                else additional_system_prompt.strip()
            )
        effective_prompt = f"{effective_prompt}\n\n{LIGHT_STATE_PROMPT}".strip()

        persistent_messages = [*history, Message(role=MessageRole.USER, content=user_input)]
        usage = ModelUsage()
        chargeable = 0
        model_calls = 0
        tool_rounds: list[ToolRound] = []
        tool_calls: list[ToolCallRecord] = []
        activated_tools: set[str] = set()
        previous_signature: str | None = None
        repeated_count = 0
        latest_observation: str | None = None
        focused_ids: tuple[str, ...] = ()
        historical_index = await self._state_store.historical_index(scope_id)
        recovered_run_id = (
            recovery_checkpoint.run_id if recovery_checkpoint is not None else None
        )
        if recovery_checkpoint is not None:
            latest_observation = json.dumps(
                {
                    "recovered_checkpoint": {
                        "phase": recovery_checkpoint.phase.value,
                        "step": recovery_checkpoint.step,
                        "pending_tool_calls": [
                            call.model_dump(mode="json")
                            for call in recovery_checkpoint.pending_tool_calls
                        ],
                        "completed_tool_results": [
                            result.model_dump(mode="json")
                            for result in recovery_checkpoint.completed_tool_results
                        ],
                    }
                },
                ensure_ascii=False,
            )
        context_session = ContextRuntime(
            memory_manager=None,
            memory_search_query=None,
            skill_context_provider=None,
            skill_store=None,
            plan_context_provider=None,
        )
        tool_event_hook = AgentEventHook(emitter)
        budget_warning_emitted = False
        force_final_answer = False
        effective_max_steps = self._max_steps

        await emitter.emit(
            AgentEventType.AGENT_STARTED,
            message=persistent_messages[-1],
            provider=resolved_provider,
            model=resolved_model,
        )
        ensure_tool_search_registered(self._tool_registry)

        for step in range(1, effective_max_steps + 1):
            decision = self._run_budget.evaluate(
                usage,
                chargeable_tokens_override=chargeable,
                model_calls_override=model_calls,
            )
            if decision.exceeded:
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_EXCEEDED,
                    step=step,
                    **run_budget_event_fields(decision, self._run_budget.config),
                )
                return self._error_result(
                    run_id,
                    persistent_messages,
                    step - 1,
                    AgentStopReason.RUN_BUDGET,
                    RunBudgetExceededError(run_budget_detail(decision)),
                    tool_rounds,
                    tool_calls,
                    usage,
                )
            if decision.should_warn and not budget_warning_emitted:
                budget_warning_emitted = True
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_WARNING,
                    step=step,
                    **run_budget_event_fields(decision, self._run_budget.config),
                )

            if self._checkpoint_store is not None:
                await self._checkpoint_store.before_model(run_id, step=step)

            active_nodes = await self._state_store.active_for_run(
                scope_id,
                run_id,
                focused_ids,
                recovered_run_id=recovered_run_id,
            )
            request_messages = (
                Message(role=MessageRole.SYSTEM, content=effective_prompt),
                Message(
                    role=MessageRole.SYSTEM,
                    name="HISTORICAL_STATE_INDEX",
                    content=render_historical_index(historical_index),
                ),
                Message(
                    role=MessageRole.SYSTEM,
                    name="ACTIVE_STATE",
                    content=render_active_state(active_nodes),
                ),
                Message(
                    role=MessageRole.SYSTEM,
                    name="LATEST_OBSERVATION",
                    content=render_observation(latest_observation),
                ),
                Message(role=MessageRole.USER, content=user_input),
            )
            tools = tuple(
                self._tool_registry.definitions_for_mode(
                    AgentMode.LIGHT,
                    activated_names=activated_tools,
                )
            )
            if decision.should_finalize or force_final_answer:
                tools = ()
                if decision.should_finalize:
                    await emitter.emit(
                        AgentEventType.RUN_BUDGET_FINALIZING,
                        step=step,
                        **run_budget_event_fields(decision, self._run_budget.config),
                    )

            await emitter.emit(
                AgentEventType.MODEL_STARTED,
                step=step,
                provider=resolved_provider,
                model=resolved_model,
                tool_schema_tokens=0,
            )
            try:
                response = await adapter.complete(
                    ModelRequest(
                        messages=request_messages,
                        model=resolved_model,
                        tools=tools,
                        max_output_tokens=output_tokens,
                    )
                )
            except Exception as exc:
                return self._error_result(
                    run_id,
                    persistent_messages,
                    step,
                    AgentStopReason.MODEL_ERROR,
                    ModelInvocationError(f"{type(exc).__name__}: {exc}"),
                    tool_rounds,
                    tool_calls,
                    usage,
                )

            usage = add_usage(usage, response.usage)
            model_calls += max(1, response.usage.model_calls)
            chargeable += chargeable_tokens(response.usage)
            envelope = normalize_content_tool_calls(
                parse_model_output(
                    response.message.content,
                    native_tool_calls=tuple(response.message.tool_calls),
                )
            )
            focused_ids = tuple(dict.fromkeys(envelope.state_focus))
            state_errors: list[str] = []
            valid_deltas = []
            for delta in envelope.state_delta:
                try:
                    validate_state_delta(delta)
                    valid_deltas.append(delta)
                except ValueError as exc:
                    state_errors.append(str(exc))
            if valid_deltas:
                try:
                    await self._state_store.apply_deltas(
                        scope_id,
                        run_id,
                        tuple(valid_deltas),
                    )
                except ValueError as exc:
                    state_errors.append(str(exc))

            native_or_fallback_calls = tuple(
                ToolCall(
                    id=call.id or f"light-call-{index + 1}",
                    name=call.name,
                    arguments=call.arguments,
                )
                for index, call in enumerate(envelope.tool_calls)
            )
            assistant_for_history = response.message.model_copy(
                update={
                    "content": None if native_or_fallback_calls else envelope.assistant_message,
                    "tool_calls": native_or_fallback_calls,
                    "reasoning": None,
                }
            )
            await emitter.emit(
                AgentEventType.MODEL_COMPLETED,
                step=step,
                provider=response.provider,
                model=response.model,
                message=assistant_for_history,
                usage=response.usage,
            )

            if not native_or_fallback_calls:
                answer = (envelope.assistant_message or "").strip()
                if answer:
                    final_message = Message(role=MessageRole.ASSISTANT, content=answer)
                    persistent_messages.append(final_message)
                    return AgentResult(
                        run_id=run_id,
                        final_message=final_message,
                        messages=tuple(persistent_messages),
                        steps=step,
                        stop_reason=AgentStopReason.FINAL_ANSWER,
                        tool_rounds=tuple(tool_rounds),
                        tool_calls=tuple(tool_calls),
                        usage=usage,
                    )
                latest_observation = (
                    "state_delta rejected: " + "; ".join(state_errors)
                    if state_errors
                    else "State updated; continue only if more evidence is needed."
                )
                persistent_messages.append(assistant_for_history)
                continue

            if decision.should_finalize or force_final_answer:
                return self._error_result(
                    run_id,
                    persistent_messages,
                    step,
                    AgentStopReason.RUN_BUDGET,
                    RunBudgetExceededError("model requested tools during finalization"),
                    tool_rounds,
                    tool_calls,
                    usage,
                )

            persistent_messages.append(assistant_for_history)
            outcome = await self._tool_round_executor.execute(
                native_or_fallback_calls,
                run_id=run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                step=step,
                mode=AgentMode.LIGHT,
                round_index=len(tool_rounds),
                closing_can_deliver=False,
                activated_tools=activated_tools,
                context_session=context_session,
                previous_signature=previous_signature,
                repeated_count=repeated_count,
                emitter=emitter,
                hook=tool_event_hook,
            )
            previous_signature = outcome.previous_signature
            repeated_count = outcome.repeated_count
            activated_tools.update(outcome.pending_activations)
            tool_calls.extend(outcome.records)
            persistent_messages.extend(outcome.result_messages)
            tool_rounds.append(
                ToolRound(
                    round_index=len(tool_rounds),
                    assistant_message=assistant_for_history,
                    records=outcome.records,
                )
            )
            latest_observation = _render_tool_observation(outcome.result_messages, state_errors)
            if outcome.repeated_error is not None:
                return self._error_result(
                    run_id,
                    persistent_messages,
                    step,
                    AgentStopReason.REPEATED_TOOL_CALL,
                    outcome.repeated_error,
                    tool_rounds,
                    tool_calls,
                    usage,
                )

            if self._max_tool_rounds is not None and len(tool_rounds) >= self._max_tool_rounds:
                # 给模型一次无工具收尾机会，不再执行新的工具。
                force_final_answer = True

        error = MaxStepsExceededError(effective_max_steps)
        return self._error_result(
            run_id,
            persistent_messages,
            effective_max_steps,
            AgentStopReason.MAX_STEPS,
            error,
            tool_rounds,
            tool_calls,
            usage,
        )

    @staticmethod
    def _error_result(
        run_id: str,
        messages: list[Message],
        step: int,
        stop_reason: AgentStopReason,
        error,
        tool_rounds: list[ToolRound],
        tool_calls: list[ToolCallRecord],
        usage: ModelUsage,
    ) -> AgentResult:
        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"[{type(error).__name__}] {error}",
        )
        if not messages or messages[-1] != final_message:
            messages = [*messages, final_message]
        return AgentResult(
            run_id=run_id,
            final_message=final_message,
            messages=tuple(messages),
            steps=max(0, step),
            stop_reason=stop_reason,
            tool_rounds=tuple(tool_rounds),
            tool_calls=tuple(tool_calls),
            usage=usage,
            error=AgentError(type=type(error).__name__, message=str(error)),
        )


def _render_tool_observation(
    messages: Sequence[Message],
    state_errors: Sequence[str],
) -> str:
    payload = [
        {
            "tool": message.name,
            "content": (message.content or "")[:8_000],
        }
        for message in messages
    ]
    if state_errors:
        payload.append({"state_errors": tuple(state_errors)})
    return json.dumps(payload, ensure_ascii=False)


__all__ = ["LightAgentLoop"]
