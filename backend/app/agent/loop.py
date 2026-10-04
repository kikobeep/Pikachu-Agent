"""单次 Run 的模型请求与工具执行循环。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from dataclasses import replace

from app.agent.budget import RunBudget, RunBudgetStatus, chargeable_tokens
from app.model.config import AgentMode, Message, MessageRole
from app.agent.context_injection import ContextRuntime
from app.agent.emitter import EventEmitter
from app.agent.error import (
    ContextPreparationError,
    ContextWindowExceededError,
    MaxStepsExceededError,
    ModelInvocationError,
)
from app.agent.events import AgentEventType
from app.agent.tool_hooks import AgentEventHook
from app.agent.tools_round_executor import ToolRoundExecutor
from app.agent.utils import offset_summary_state
from app.checkpoint.config import RunCheckpoint
from app.checkpoint.store import CheckpointStore
from app.context.handoff import HandoffSnapshot, collect_git_snapshot
from app.context.handoff_store import HandoffStore
from app.context.manager import ContextManager
from app.context.summary import ConversationSummaryState
from app.context.task_boundary import (
    TaskBoundaryTrigger,
    basis_message_id,
    classify as classify_task_boundary,
    ensure_initial_task,
    observe as observe_task_boundary,
)
from app.memory.manager import MemoryManager
from app.memory.search import recent_user_message_texts
from app.memory.search_config import MemorySearchInputs
from app.model.config import ModelProvider, ModelRequest, ModelUsage
from app.model.registry import ModelAdapterRegistry
from app.skills.context import SkillContextProvider
from app.skills.store import SkillStore
from app.plan.context import PlanContextProvider
from app.plan.runner import PlanRunner
from app.tools.executor import ToolExecutor
from app.tools.register import ToolRegistry, ensure_tool_search_registered

from .config import (
    _EMPTY_FINAL_RETRY_MESSAGE,
    _RUN_BUDGET_CLOSING_MESSAGE,
    _RUN_BUDGET_FINALIZATION_MESSAGE,
    _RUN_BUDGET_WARNING_MESSAGE,
    _TEXTUAL_TOOL_CALL_RETRY_MESSAGE,
)
from .error import AgentRuntimeError, RunBudgetExceededError
from .result import AgentError, AgentResult, AgentStopReason, ToolCallRecord, ToolRound
from .utils import (
    add_usage,
    run_budget_detail,
    run_budget_event_fields,
    usage_call_count,
    provider_name,
    RequestPrefixState,
    is_tool_call_text
)
_PLAN_MODE_SYSTEM_MESSAGE = (
    "你现在处于 PLAN MODE（计划执行模式）：按照当前活动计划的 current_step 逐步"
    "分析、调查和推进计划；不要修改用户环境。\n"
    "对于用户提出的新规划或多步骤目标，必须先调用 plan_create 创建一个 Plan，"
    "至少包含 title、goal 和具体 steps；plan_create 成功前不得直接输出最终计划。"
    "不要为了确认是否存在计划而先调用 plan_list；plan_list 只用于用户要求查看"
    "已有计划，或明确要求继续已有计划的情况。\n"
    "默认不要调用 memory_search。只有用户明确要求使用记忆，或当前计划确实依赖"
    "已保存的个人偏好、历史决定或长期事实时，才调用 memory_search。涉及今天、"
    "当前日期或相对时间时，可以调用 get_current_time。\n"
    "你可以使用只读 / 搜索工具（read_file、list_files、web_search、get_current_time、"
    "memory_read、memory_search、history_search/read、evidence_search/read）以及计划工具"
    "（plan_create、plan_update、plan_get、plan_list）。\n"
    "每次只处理上下文中展示的 current_step，可以调用多个工具；当前步骤获得充分"
    "证据后，使用 plan_update 将它标记为 done，并填写完成依据；无法继续时标记为"
    "blocked 并填写原因。不要主动执行后续未展示的步骤，也不要伪造完成状态。\n"
    "plan_update 只用于更新已有步骤的状态和备注；标记当前步骤状态时传入"
    "step_id、step_status 和必要的 step_note，不要尝试重建或替换整个步骤列表。\n"
    "只有当前步骤完成或阻塞后，才能进入下一个步骤；所有步骤完成后再返回总结。"
)

_TOOL_ROUND_LIMIT_FALLBACK_MESSAGE = (
    "已达到本 Run 的工具调用轮次上限，系统已停止继续执行工具。已有工具结果"
    "仍保留在本轮记录中，但模型未能在无工具模式下生成可靠总结；如需继续，"
    "请基于当前结果提出下一步要求。"
)

class AgentLoop:
    def __init__(
        self,
        *,
        model_registry: ModelAdapterRegistry,
        provider: ModelProvider | str | None,
        model: str | None,

        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,

        system_prompt: str | None,
        max_steps: int,
        max_tool_rounds: int | None,
        max_output_tokens: int | None,

        context_manager: ContextManager,
        plan_context_provider: PlanContextProvider | None,
        plan_runner: PlanRunner | None,
        checkpoint_store: CheckpointStore | None,
        memory_manager: MemoryManager | None,
        memory_auto_search_enabled: bool = True,
        skill_store: SkillStore | None,
        skill_context_provider: SkillContextProvider | None,

        run_budget: RunBudget,
        handoff_store: HandoffStore | None = None,
        workspace_root: str | Path | None = None,
    ):
        self._model_registry = model_registry
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor #执行单个调用
        self._tool_round_executor = ToolRoundExecutor( # 组织一整轮的工具调用，可能包含多次tool call
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
        self._context_manager = context_manager
        self._plan_context_provider = plan_context_provider
        self._plan_runner = plan_runner
        self._checkpoint_store = checkpoint_store
        self._memory_manager = memory_manager
        self._memory_auto_search_enabled = memory_auto_search_enabled
        self._skill_store = skill_store
        self._skill_context_provider = skill_context_provider
        self._run_budget = run_budget
        self._handoff_store = handoff_store
        self._workspace_root = workspace_root

    @staticmethod
    def _error_message(error: AgentRuntimeError) -> Message:
        import traceback
        print("!! _error_message called with:", repr(error))
        traceback.print_stack()
        return Message(
            role=MessageRole.ASSISTANT,
            content=(
                f"[{type(error).__name__}] {error}\n\n"
                "本次 Run 因内部错误结束，可稍后重试或调整输入。\n\n"
                f"TRACEBACK:\n{traceback.format_exc()[-1500:]}"
            ),
        )

    # AgentEventHook-> 工具执行事件通知对象
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

        tool_event_hook = AgentEventHook(emitter)
        user_message = Message(role = MessageRole.USER, content = user_input)
        messages = [*history, user_message]
        usage = ModelUsage()
        sys_already_exits = False
        request_system_message = None
        current_summary_state = summary_state
        repair_instruction: Message | None = None
        budget_warning_emitted = False
        budget_closing_delivery_used = False
        budget_closing_report_used = False
        budget_closing_started = False
        budget_closing_delivery_used = False
        empty_response_retry_used = False
        tool_call_text_retry_used = False

        effective_system_prompt = self._system_prompt
        if additional_system_prompt and additional_system_prompt.strip():
            effective_system_prompt = (
                f"{self._system_prompt}\n\n{additional_system_prompt.strip()}"
                if self._system_prompt
                else additional_system_prompt.strip()
            )

        if effective_system_prompt:
            for record in history:
                if record.role is MessageRole.SYSTEM and record.content == effective_system_prompt:
                    sys_already_exits = True
                    break
        
        if effective_system_prompt is not None and not sys_already_exits: # history 里没有system_prompt
            request_system_message = Message(role=MessageRole.SYSTEM, content=effective_system_prompt)
        
        # 历史消息的长度 = 原始历史消息 + 加入system_prompt的偏移
        history_message_length = len(history)
        history_message_offset = 0
        if request_system_message:
            history_message_offset = 1
        history_message_length = (
            history_message_length + history_message_offset
        )

        context_session = ContextRuntime(
            memory_manager=(
                self._memory_manager
                if self._memory_auto_search_enabled
                else None
            ),
            # 初始化：用户输入 + 最近消息 + 摘要目标
            #        ↓
            # 首次加载记忆：补上任务标题和进行中步骤
            #        ↓
            # 用完整查询检索记忆
            memory_search_query=(
                MemorySearchInputs(
                    user_message=user_input,
                    recent_user_messages=recent_user_message_texts(history),
                    summary_goal=(
                        summary_state.summary.goal
                        if summary_state is not None
                        and summary_state.summary.goal
                        else None
                    ),
                )
                if self._memory_manager is not None
                else None
            ),
            skill_store=self._skill_store,
            skill_context_provider=self._skill_context_provider,
            plan_context_provider=self._plan_context_provider,
        )

        await emitter.emit(
            AgentEventType.AGENT_STARTED,
            message=user_message,
            provider=provider_name(self._provider),
            model=self._model,
        )

        async def stop_with_error(
            error: AgentRuntimeError,
            stop_reason: AgentStopReason,
            *,
            step: int,
        ) -> AgentResult:
            """构造失败结果；公共 run() 在后置阶段结束后发射终止事件。"""

            return self._result(
                run_id=run_id,
                final_message=self._error_message(error),
                messages=messages,
                steps=step,
                stop_reason=stop_reason,
                tool_rounds=tool_rounds,
                tool_calls=tool_calls,
                usage=usage,
                error=error,
                summary_state=current_summary_state,
            )
        
        def stop_at_tool_round_limit(*, step: int) -> AgentResult:
            """模型拒绝无工具收尾时，由 Harness 给出诚实且确定性的边界说明。"""

            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=_TOOL_ROUND_LIMIT_FALLBACK_MESSAGE,
            )
            messages[-1] = final_message
            return self._result(
                run_id=run_id,
                final_message=final_message,
                messages=messages,
                steps=step,
                stop_reason=AgentStopReason.FINAL_ANSWER,
                tool_rounds=tool_rounds,
                tool_calls=tool_calls,
                usage=usage,
                summary_state=current_summary_state,
            )

        ensure_tool_search_registered(self._tool_registry) # 如果有按需加载的tool -> 必须注册toolsearch工具
        
        needs_extra_answer_step = False
        main_model_calls = 0
        budget_chargeable_tokens = 0
        tool_rounds: list[ToolRound] = []
        tool_calls: list[ToolCallRecord] = []
        request_prefix_state: RequestPrefixState | None = None
        previous_signature: str | None = None
        repeated_count = 0
        activated_tools: set[str] = set()

        HANDOFF_STEP_BONUS = 5
        effective_max_steps = self._max_steps
        handoff_used = False
        step = 0

        # 每个新的用户 Run 在可见主请求前观察一次任务边界。首条消息只初始化
        # active hash；后续消息使用隐藏分类请求，并通过稳定窗口确认切换。
        task_trigger = TaskBoundaryTrigger.AUTO
        try:
            first_user_content = next(
                (message.content for message in history if message.role is MessageRole.USER),
                user_input,
            )
            first_basis = basis_message_id(
                conversation_id=conversation_id,
                history=(),
                user_input=first_user_content,
            )
            current_summary_state = ensure_initial_task(
                current_summary_state,
                conversation_id=conversation_id,
                basis=first_basis,
            )
            if history:
                boundary_adapter = self._model_registry.get(self._provider)
                boundary_model = self._model or boundary_adapter.default_model
                boundary_basis = basis_message_id(
                    conversation_id=conversation_id,
                    history=history,
                    user_input=user_input,
                )
                boundary_decision, boundary_usage = await classify_task_boundary(
                    boundary_adapter,
                    model=boundary_model,
                    history=history,
                    user_input=user_input,
                    summary_state=current_summary_state,
                )
                current_summary_state, observation = observe_task_boundary(
                    current_summary_state,
                    decision=boundary_decision,
                    conversation_id=conversation_id,
                    basis=boundary_basis,
                    required_stable_count=self._context_manager.task_boundary_required_stable_count,
                )
                if observation.confirmed_change:
                    task_trigger = TaskBoundaryTrigger.TASK
                usage = add_usage(usage, boundary_usage)
                main_model_calls += usage_call_count(boundary_usage)
                budget_chargeable_tokens += chargeable_tokens(boundary_usage)
        except Exception:
            # 边界判断不能阻断主任务；失败时保持当前任务并走 AUTO。
            task_trigger = TaskBoundaryTrigger.AUTO

        # Sequential Plan Execution 只在 PLAN MODE 启用；普通模式继续由模型
        # 自主参考计划，不由 PlanRunner 自动推进步骤。
        if self._plan_runner is not None and mode is AgentMode.PLAN:
            try:
                await self._plan_runner.on_run_started(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    emitter=emitter,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "plan runner failed while restoring the current plan"
                )

        while True:
            step += 1
            if step > effective_max_steps + 1:
                break

            is_extra_answer_step = step > effective_max_steps
            if is_extra_answer_step and not needs_extra_answer_step: # needs_extra_answer_step：是否允许超出正常步数，再执行一次收尾
                break
            
            # 计算模型目前的token消耗量以及model call次数是否超标
            budget_decision = self._run_budget.evaluate(
                usage,
                chargeable_tokens_override=budget_chargeable_tokens,
                model_calls_override=main_model_calls,
            )

            budget_warning_in_request = budget_decision.should_warn
            budget_config = self._run_budget.config
            # 超出硬限制
            if budget_decision.exceeded:
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_EXCEEDED,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
                return await stop_with_error(
                    RunBudgetExceededError(
                        run_budget_detail(budget_decision)
                    ),
                    AgentStopReason.RUN_BUDGET,
                    step=max(0, step - 1),
                )
            if budget_decision.should_warn and not budget_warning_emitted:
                budget_warning_emitted = True
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_WARNING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )

            # 应停止扩展任务，准备交付回答
            budget_requires_closing = budget_decision.should_finalize
            if budget_decision.should_finalize:
                if budget_closing_delivery_used: # 如果交付工具轮已经用过，接下来应该让模型汇报结果，不再继续扩展工作
                    if budget_closing_report_used: # 如果最终汇报机会也已经用过，却又进入了下一步，就结束运行并报告预算错误
                        await emitter.emit(
                            AgentEventType.RUN_BUDGET_EXCEEDED,
                            step=step,
                            **run_budget_event_fields(
                                budget_decision,
                                budget_config,
                                status=RunBudgetStatus.EXCEEDED,
                            ),
                        )
                        return await stop_with_error(
                            RunBudgetExceededError(
                                "dedicated closing report call was already used"
                            ),
                            AgentStopReason.RUN_BUDGET,
                            step=max(0, step - 1),
                        )
                    budget_closing_report_used = True
                elif not budget_closing_started: # 如果还没有用交付工具轮，而且是首次进入收尾
                    budget_closing_started = True
                    await emitter.emit(
                        AgentEventType.RUN_BUDGET_FINALIZING,
                        step=step,
                        **run_budget_event_fields(budget_decision, budget_config),
                    )
                
            if self._checkpoint_store is not None:
                await self._checkpoint_store.before_model(run_id,step = step)
                

            tool_round_limit_reached = (
                self._max_tool_rounds is not None
                and len(tool_rounds) >= self._max_tool_rounds
            )
            # 只要满足任意一个条件，就要求直接收尾
            must_stop_tool_calls = (
                is_extra_answer_step or tool_round_limit_reached
            )
            # 判断能否再给一轮收尾工具机会：预算已经要求开始收尾 and 还没用过收尾工具轮 and 没有其他原因要求立即停止工具调用
            can_use_closing_tools = (
                budget_requires_closing
                and not budget_closing_delivery_used
                and not must_stop_tool_calls
            )
            # 两种情况下要求最终回答：有预算之外的强制结束原因 or 预算要求收尾，而且已经不能再给收尾工具机会
            must_answer_now = must_stop_tool_calls or (
                budget_requires_closing and not can_use_closing_tools
            )

            # message
            raw_source_messages = tuple(message for message in messages)
            if repair_instruction is not None:
                raw_source_messages = (
                    *raw_source_messages,
                    repair_instruction,
                )
            source_messages = (
                (request_system_message, *raw_source_messages)
                if request_system_message is not None
                else raw_source_messages
            )
            # prepare() 处理的是请求列表，所以传入时要 +1。处理完成后，保存回 current_summary_state 的位置需要重新对应原始历史要 -1
            request_summary_state = offset_summary_state(
                current_summary_state,
                history_message_offset,
            )
            request_messages = source_messages
            # tool
            if can_use_closing_tools:
                request_tools = self._tool_registry.closing_definitions_for_mode(
                    mode,
                    activated_names=activated_tools,
                )
                if not request_tools:
                    can_use_closing_tools = False
                    must_answer_now = True
            elif must_answer_now:
                request_tools = ()
            else:
                request_tools = self._tool_registry.definitions_for_mode(
                    mode,
                    activated_names=activated_tools,
                )
                
            # model：先解析实际使用的模型和输出上限，确保预算与请求完全一致。
            try:
                adapter = self._model_registry.get(self._provider)
                resolved_model = self._model or adapter.default_model
                resolved_provider = adapter.provider
                effective_max_output_tokens = (
                    self._max_output_tokens or adapter.config.default_max_output_tokens
                )
                if budget_requires_closing and must_answer_now:
                    effective_max_output_tokens = min(
                        effective_max_output_tokens,
                        budget_config.finalization_max_output_tokens,
                    )
            except Exception as exc:
                import traceback
                print("BUDGET_ERROR:", traceback.format_exc())
                return await stop_with_error(
                    ModelInvocationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )
            

            # 为本次模型请求补充上下文 
            try:
                # 易变事实通过按需工具获取；Run 级缓存与 Skill 状态由
                # RuntimeContextSession 管理，Plan 仍在每个 Step 重新读取。
                trailing_system_messages: list[Message] = []

                context_injection = await context_session.build(
                    conversation_id=conversation_id,
                    recovery_checkpoint=recovery_checkpoint
                )
                context_messages = list(context_injection.messages)
                if mode is AgentMode.PLAN:
                    context_messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            name="plan_mode",
                            content=_PLAN_MODE_SYSTEM_MESSAGE,
                        )
                    )

                if budget_warning_in_request:
                    context_messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_RUN_BUDGET_WARNING_MESSAGE,
                        )
                    )
                context_messages = tuple(context_messages)
                plan_context_messages = tuple(
                    message
                    for message in context_messages
                    if message.name == "PLAN CONTEXT"
                )
                other_context_messages = tuple(
                    message
                    for message in context_messages
                    if message.name != "PLAN CONTEXT"
                )
                if other_context_messages:
                    request_messages = (
                        *request_messages[:history_message_length],
                        *other_context_messages,
                        *request_messages[history_message_length:],
                    )
                # Plan 是易变状态。把最新快照放在上一轮工具结果之后，避免模型
                # 采用历史 tool result 中已经过期的 revision。
                if plan_context_messages:
                    request_messages = (
                        *request_messages,
                        *plan_context_messages,
                    )
                
                if can_use_closing_tools:
                    request_messages = (
                        *request_messages,
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_RUN_BUDGET_CLOSING_MESSAGE,
                        ),
                    )
                elif must_answer_now:
                    if budget_requires_closing:
                        final_instruction = _RUN_BUDGET_FINALIZATION_MESSAGE
                    else:
                        final_instruction = (
                            "工具调用轮次已用完。请读取最后一条工具结果，停止"
                            "调用工具并直接回答用户。对于 verification_status="
                            "unverified 的电脑操作，只能说明事件已投递、效果未"
                            "确认，不能宣称界面操作已经完成。"
                        )
                    request_messages = (
                        *request_messages,
                        Message(
                            role=MessageRole.SYSTEM,
                            content=final_instruction,
                        ),
                    )
            except Exception as exc:
                return await stop_with_error(
                    ContextPreparationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )

            # 上下文检查，看能否复用，cache数量
            continuation_messages = (
                request_prefix_state.extend(
                    source_messages=source_messages,
                    context_messages=context_messages,
                    tools=request_tools,
                )
                if request_prefix_state is not None and not must_answer_now #考虑到 force_final_answer在request_messages后面增加了instruction，但并没有在source_messages后增加，所以需要特殊处理
                else None
            )
            
            context_input_messages = continuation_messages or request_messages
            cache_prefix_reused = continuation_messages is not None
            cache_prefix_message_count = ( # 传0，避免把旧摘要、临时记忆或当前工具结果误当作原始历史处理
                len(request_prefix_state.sent_messages)
                if cache_prefix_reused and request_prefix_state is not None
                else 0
            )

            context_history_count = (
                0
                if continuation_messages is not None
                else history_message_length
            )
            
            context_summary_state = (
                None if continuation_messages is not None else request_summary_state
            )

            try:
                context_decision = await self._context_manager.prepare( # 先正常准备本轮上下文
                    context_input_messages,
                    tools=request_tools,
                    model=resolved_model,
                    provider=resolved_provider,
                    max_output_tokens=effective_max_output_tokens,
                    history_count=context_history_count,
                    summary_state=context_summary_state,
                    trigger=task_trigger,
                )
                task_trigger = TaskBoundaryTrigger.AUTO
                # 正在复用上下文，但它已经太长，而且需要首次生成历史摘要
                if continuation_messages is not None and (context_decision.exceeds_input_budget or (current_summary_state is None and context_decision.requires_compaction and context_decision.needs_next_compaction_stage)): 
                    continuation_messages = None
                    cache_prefix_reused = False
                    cache_prefix_message_count = 0
                    context_decision = await self._context_manager.prepare(
                        request_messages,
                        tools=request_tools,
                        model=resolved_model,
                        provider=resolved_provider,
                        max_output_tokens=effective_max_output_tokens,
                        history_count=history_message_length,
                        summary_state=request_summary_state,
                        trigger=TaskBoundaryTrigger.AUTO,
                    )
            except Exception as exc:
                return await stop_with_error(
                    ContextPreparationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )
            
            handoff_completed = False
            if context_decision.needs_handoff and not handoff_used:
                handoff_used = True
                previous_usage = context_decision.summary_usage
                try:
                    handoff_decision = await self._context_manager.prepare(
                        (
                            *source_messages, *context_messages,
                            *request_messages[len(source_messages) + len(context_messages):],
                        ),
                        tools=request_tools,
                        model=resolved_model,
                        provider=resolved_provider,
                        max_output_tokens=effective_max_output_tokens,
                        history_count=len(messages) + history_message_offset,
                        summary_state=context_decision.summary_state,
                        handoff=True,
                    )
                except Exception as exc:
                    return await stop_with_error(
                        ContextPreparationError(f"handoff failed: {exc}"),
                        AgentStopReason.CONTEXT_ERROR, step=step,
                    )
                context_decision = replace(
                    handoff_decision,
                    summary_usage=add_usage(previous_usage, handoff_decision.summary_usage),
                )
                continuation_messages = None
                request_prefix_state = None
                cache_prefix_reused = False
                cache_prefix_message_count = 0
                handoff_completed = (
                    not context_decision.exceeds_input_budget
                    and context_decision.summary_state is not None
                )
                if handoff_completed:
                    effective_max_steps += HANDOFF_STEP_BONUS
                    # 摘要覆盖位置始终对应原始历史。
                    history_message_length = len(messages) + history_message_offset

            if continuation_messages is not None:
                context_decision = replace(
                    context_decision,
                    summary_state=request_summary_state,
                )
                previous_prefix = request_prefix_state.sent_messages
                cache_prefix_reused = (
                    context_decision.messages[: len(previous_prefix)]
                    == previous_prefix
                )
                if not cache_prefix_reused:
                    cache_prefix_message_count = 0

            current_summary_state = offset_summary_state(
                context_decision.summary_state,
                -history_message_offset,
            )

            usage = add_usage(usage, context_decision.summary_usage)
            main_model_calls += usage_call_count(context_decision.summary_usage) 
            budget_chargeable_tokens += chargeable_tokens( # uncached + output tokens
                context_decision.summary_usage
            )
            request_messages = context_decision.messages
            request_tools = context_decision.tools

            # 计算summary后的预算
            budget_decision = self._run_budget.evaluate(
                usage, chargeable_tokens_override=budget_chargeable_tokens,
                model_calls_override=main_model_calls,
            )
            if budget_decision.exceeded:
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_EXCEEDED,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
                return await stop_with_error(
                    RunBudgetExceededError(run_budget_detail(budget_decision)),
                    AgentStopReason.RUN_BUDGET,
                    step=max(0, step - 1),
                )
            if budget_decision.should_warn and not budget_warning_emitted:
                budget_warning_emitted = True
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_WARNING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
            if budget_decision.should_warn and not budget_warning_in_request: # prepare后，预算到达警告线，需要补上警告
                warning_appended_after_prepare = True
                request_messages = (
                    *request_messages,
                    Message(
                        role=MessageRole.SYSTEM,
                        content=_RUN_BUDGET_WARNING_MESSAGE,
                    ),
                )
            else:
                warning_appended_after_prepare = False

            if budget_decision.should_finalize and not budget_requires_closing: # prepare达到收尾线，需要调整即将发送的请求
                budget_requires_closing = True
                budget_closing_started = True
                can_use_closing_tools = not must_stop_tool_calls
                if can_use_closing_tools:
                    request_tools = (
                        self._tool_registry.closing_definitions_for_mode(
                            mode,
                            activated_names=activated_tools,
                        )
                    )
                    can_use_closing_tools = bool(request_tools)
                must_answer_now = must_stop_tool_calls or not can_use_closing_tools
                if must_answer_now:
                    request_tools = ()
                    effective_max_output_tokens = min(
                        effective_max_output_tokens,
                        budget_config.finalization_max_output_tokens,
                    )
                request_messages = (
                    *request_messages,
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            _RUN_BUDGET_CLOSING_MESSAGE
                            if can_use_closing_tools
                            else _RUN_BUDGET_FINALIZATION_MESSAGE
                        ),
                    ),
                )
                await emitter.emit(
                    AgentEventType.RUN_BUDGET_FINALIZING,
                    step=step,
                    **run_budget_event_fields(budget_decision, budget_config),
                )
            
            if handoff_completed:
                # 原始历史保持完整，摘要由 ConversationService 持久化。
                # 此快照用于诊断，后续 Run 仍从 SummaryStore 恢复摘要。
                handoff_snapshot = HandoffSnapshot(
                    **current_summary_state.summary.model_dump(),
                    git_state=await collect_git_snapshot(self._workspace_root),
                    conversation_id=conversation_id or run_id,
                    summary_covered_message_count=current_summary_state.covered_message_count,
                    handoff_reason="普通压缩后仍超限，已完成扩大范围的摘要",
                    step_at_handoff=step,
                    model_calls_at_handoff=main_model_calls,
                )
                try:
                    if self._handoff_store is not None:
                        await self._handoff_store.save(handoff_snapshot)
                except Exception:
                    logging.getLogger(__name__).exception("handoff snapshot save failed")
                await emitter.emit(AgentEventType.CONTEXT_HANDOFF, step=step)


            if context_decision.exceeds_input_budget:
                return await stop_with_error(
                    ContextWindowExceededError(
                        context_decision.estimated_input_tokens or 0,
                        context_decision.input_budget or 0,
                    ),
                    AgentStopReason.CONTEXT_ERROR,
                    step=step,
                )
                
                 
            if request_prefix_state is not None  and request_tools != request_prefix_state.tools:
        
                cache_prefix_reused = False
                cache_prefix_message_count = 0

            
            # 保存本轮准备发送的请求，供下一次循环尝试复用消息前缀
            if not must_answer_now and not warning_appended_after_prepare:
                request_prefix_state = RequestPrefixState(
                    source_messages=source_messages,
                    context_messages=context_messages,
                    tools=request_tools,
                    sent_messages=request_messages,
                )
            else:
                request_prefix_state = None

            # model request
            try:
                async def emit_text_delta(delta: str) -> None:
                    if not delta:
                        return
                    await emitter.emit(
                        AgentEventType.MODEL_OUTPUT_DELTA,
                        step=step,
                        provider=resolved_provider,
                        model=resolved_model,
                        delta=delta,
                    )

                response = await adapter.complete_stream(
                    ModelRequest(
                        messages=request_messages,
                        model=resolved_model,
                        tools=request_tools,
                        max_output_tokens=effective_max_output_tokens,
                    ),
                    on_text_delta=emit_text_delta,
                    on_reasoning_delta=None,
                )
            except Exception as exc:
                import traceback
                print("STREAM_ERROR:", traceback.format_exc())
                logging.getLogger(__name__).exception(
                    "model stream failed at step %s", step,
                )
                return await stop_with_error(
                    ModelInvocationError(f"{type(exc).__name__}: {exc}"),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )

            usage = add_usage(usage, response.usage)
            main_model_calls += max(1, response.usage.model_calls)
            budget_chargeable_tokens += chargeable_tokens(response.usage)
            assistant_message = response.message.model_copy(
                update={"reasoning": None}
            )
            repair_instruction = None
            messages.append(assistant_message)
            await emitter.emit(
                AgentEventType.MODEL_COMPLETED,
                step=step,
                provider=response.provider,
                model=response.model,
                message=assistant_message,
                usage=response.usage,
            )

            tool_calls_in_message = assistant_message.tool_calls
            if must_answer_now and tool_calls_in_message:
                if budget_requires_closing:
                    return await stop_with_error(
                        RunBudgetExceededError(
                            "model attempted a tool call during budget finalization"
                        ),
                        AgentStopReason.RUN_BUDGET,
                        step=step,
                    )
                if tool_round_limit_reached:
                    return stop_at_tool_round_limit(step=step)
                return await stop_with_error(
                    ModelInvocationError(
                        "model attempted a tool call during forced finalization"
                    ),
                    AgentStopReason.MODEL_ERROR,
                    step=step,
                )
            if not tool_calls_in_message:
                if not (assistant_message.content or "").strip():
                    if tool_round_limit_reached:
                        return stop_at_tool_round_limit(step=step)
                 
                    messages.pop()
                    if budget_requires_closing:
                        return await stop_with_error(
                            RunBudgetExceededError(
                                "model returned empty content during budget "
                                "finalization"
                            ),
                            AgentStopReason.RUN_BUDGET,
                            step=step,
                        )
                    if must_answer_now:
                        return await stop_with_error(
                            ModelInvocationError(
                                "model returned empty content during forced "
                                "finalization"
                            ),
                            AgentStopReason.MODEL_ERROR,
                            step=step,
                        )
                    # 模型既没有返回正文，也没有发起工具调用。代码允许补救一次
                    if not empty_response_retry_used:
                        empty_response_retry_used = True
                        repair_instruction = Message(
                            role=MessageRole.SYSTEM,
                            content=_EMPTY_FINAL_RETRY_MESSAGE,
                        )
                        continue
                    return await stop_with_error(
                        ModelInvocationError(
                            "model returned empty content twice without tool calls"
                        ),
                        AgentStopReason.MODEL_ERROR,
                        step=step,
                    )
                
                if is_tool_call_text(assistant_message.content):
                    if must_answer_now:
                        if budget_requires_closing:
                            return await stop_with_error(
                                RunBudgetExceededError(
                                    "model emitted a textual tool call during "
                                    "budget finalization"
                                ),
                                AgentStopReason.RUN_BUDGET,
                                step=step,
                            )
                        if tool_round_limit_reached:
                            return stop_at_tool_round_limit(step=step)
                        return await stop_with_error(
                            ModelInvocationError(
                                "model emitted a textual tool call during forced "
                                "finalization"
                            ),
                            AgentStopReason.MODEL_ERROR,
                            step=step,
                        )
                    messages.pop()
                    if not tool_call_text_retry_used:
                        tool_call_text_retry_used = True
                        repair_instruction = Message(
                            role=MessageRole.SYSTEM,
                            content=_TEXTUAL_TOOL_CALL_RETRY_MESSAGE,
                        )
                        continue
                    return await stop_with_error(
                        ModelInvocationError(
                            "model emitted a textual tool call twice without "
                            "structured tool_calls"
                        ),
                        AgentStopReason.MODEL_ERROR,
                        step=step,
                    )
                final_message = assistant_message
                return self._result(
                    run_id=run_id,
                    final_message=final_message,
                    messages=messages,
                    steps=step,
                    stop_reason=AgentStopReason.FINAL_ANSWER,
                    tool_rounds=tool_rounds,
                    tool_calls=tool_calls,
                    usage=usage,
                    summary_state=current_summary_state,
                )

            round_outcome = await self._tool_round_executor.execute(
                tool_calls_in_message,
                run_id=run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                step=step,
                mode=mode,
                round_index=len(tool_rounds),
                closing_can_deliver=can_use_closing_tools,
                activated_tools=activated_tools,
                context_session=context_session,
                previous_signature=previous_signature,
                repeated_count=repeated_count,
                emitter=emitter,
                hook=tool_event_hook,
            )

            # 保存工具执行记录，以及下一次请求模型需要的工具结果消息。
            tool_calls.extend(round_outcome.records)
            messages.extend(round_outcome.result_messages)

        
            previous_signature = round_outcome.previous_signature
            repeated_count = round_outcome.repeated_count

            if self._plan_runner is not None and mode is AgentMode.PLAN:
                try:
                    if round_outcome.plan_created and round_outcome.plan_id:
                        await self._plan_runner.on_plan_created(
                            conversation_id=conversation_id,
                            plan_id=round_outcome.plan_id,
                            run_id=run_id,
                            emitter=emitter,
                        )
                    else:
                        await self._plan_runner.on_tool_round_finished(
                            conversation_id=conversation_id,
                            run_id=run_id,
                            plan_id=round_outcome.plan_id,
                            emitter=emitter,
                        )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "plan runner failed after a tool round"
                    )

            if round_outcome.repeated_error is not None:
                return self._result(
                    run_id=run_id,
                    final_message=self._error_message(round_outcome.repeated_error),
                    messages=messages,
                    steps=step,
                    stop_reason=AgentStopReason.REPEATED_TOOL_CALL,
                    tool_rounds=tool_rounds,
                    tool_calls=tool_calls,
                    usage=usage,
                    error=round_outcome.repeated_error,
                    summary_state=current_summary_state,
                )

            # 保存本轮搜索到、待激活的工具名称。
            pending_activations = set(round_outcome.pending_activations)

            # 将这次模型回复发起的所有工具调用，记为一轮。
            tool_rounds.append(
                ToolRound(
                    round_index=len(tool_rounds),
                    assistant_message=assistant_message,
                    records=tuple(round_outcome.records),
                )
            )
            # 待激活工具加入 activated_tools
            activated_tools.update(pending_activations)

            needs_extra_answer_step = (
                step == effective_max_steps
                and self._run_budget.evaluate(
                    usage,
                    chargeable_tokens_override=budget_chargeable_tokens,
                    model_calls_override=main_model_calls,
                ).should_finalize
            )

        error = MaxStepsExceededError(effective_max_steps)
        return self._result(
            run_id=run_id,
            final_message=self._error_message(error),
            messages=messages,
            steps=effective_max_steps,
            stop_reason=AgentStopReason.MAX_STEPS,
            tool_rounds=tool_rounds,
            tool_calls=tool_calls,
            usage=usage,
            error=error,
            summary_state=current_summary_state,
        )
    
    @staticmethod
    def _result(
        *,
        run_id: str,
        final_message: Message,
        messages: Sequence[Message],
        steps: int,
        stop_reason: AgentStopReason,
        tool_rounds: list[ToolRound],
        tool_calls: list[ToolCallRecord],
        usage: ModelUsage,
        error: AgentRuntimeError | None = None,
        summary_state: ConversationSummaryState | None = None,
        plan_id: str | None = None,
    ) -> AgentResult:
        complete_messages = tuple(messages)
        if not complete_messages or complete_messages[-1] != final_message:
            complete_messages = (*complete_messages, final_message)

        return AgentResult(
            run_id=run_id,
            final_message=final_message,
            messages=complete_messages,
            steps=steps,
            stop_reason=stop_reason,
            tool_rounds=tuple(tool_rounds),
            tool_calls=tuple(tool_calls),
            usage=usage,
            error=(
                AgentError(type=type(error).__name__, message=str(error))
                if error is not None
                else None
            ),
            summary_state=summary_state,
            plan_id=plan_id,
        )

'''
进入 run()
│
├─ 1. 初始化本次 Run
│    ├─ 组合历史消息和当前用户消息
│    ├─ 判断是否需要补充系统提示
│    ├─ 计算历史消息边界及系统消息偏移
│    ├─ 创建 context_session
│    ├─ 初始化工具调用记录、工具轮次、已激活工具集合
│    ├─ 初始化 usage、预算计数和 summary_state
│    ├─ 初始化空回复、文本工具调用的重试标记
│    └─ request_prefix_state = None
│
└─ 2. 进入 Step 循环 ◄────────────────────────────────────┐
     │                                                   │
     ├─ 是否超过正常步数？                               │
     │    ├─ 否 → 继续                                   │
     │    └─ 是                                          │
     │         ├─ needs_extra_answer_step=True → 进入收尾步  │
     │         └─ 否 → 退出循环，返回步数超限错误          │
     │                                                   │
     ├─ 3. 第一次预算检查：run_budget.evaluate()           │
     │    │                                              │
     │    ├─ exceeded=True？                              │
     │    │    └─ 是 → 发出预算超限事件，结束 Run          │
     │    │                                              │
     │    ├─ should_warn=True？                           │
     │    │    ├─ 标记本轮请求需要预算警告                 │
     │    │    └─ 尚未报告过警告 → 发出警告事件           │
     │    │                                              │
     │    └─ should_finalize=True？                       │
     │         ├─ budget_requires_closing = True              │
     │         ├─ 首次收尾 → 发出进入收尾事件             │
     │         └─ 收尾交付机会已经用过？                  │
     │              ├─ 否 → 尝试保留一次收尾工具机会      │
     │              └─ 是                                │
     │                   ├─ 汇报机会未用 → 本轮最终汇报   │
     │                   └─ 汇报机会已用 → 结束 Run       │
     │                                                   │
     ├─ 4. 确定本轮能否调用工具                           │
     │    ├─ 检查工具轮次是否达到上限                     │
     │    ├─ must_stop_tool_calls                       │
     │    │    = 步数收尾 / 工具轮次上限 / 其他停止条件   │
     │    ├─ can_use_closing_tools                          │
     │    │    = 预算要求收尾，而且还有收尾工具机会       │
     │    └─ must_answer_now                           │
     │         = 其他限制要求结束                         │
     │           或「预算要求收尾且不能再交付」           │
     │                                                   │
     ├─ 5. 准备消息、工具和模型配置                       │
     │    ├─ 构建 source_messages                        │
     │    │    └─ 有 repair_instruction → 加入       │
     │    ├─ 选择 request_tools                          │
     │    │    ├─ 正常执行 → 当前允许且已激活的工具       │
     │    │    ├─ 收尾交付 → 只提供收尾工具               │
     │    │    └─ 强制回答 → 不提供工具                   │
     │    ├─ 解析 adapter、model、输出 token 上限         │
     │    └─ context_session.build()                     │
     │         ├─ 加载或复用本 Run 的记忆上下文           │
     │         ├─ 加入技能目录和已激活技能                 │
     │         ├─ 加入当前任务和恢复检查点                 │
     │         └─ 加入预算警告、收尾等必要提示             │
     │                                                   │
     ├─ 6. 尝试复用上一轮消息前缀                         │
     │    │                                              │
     │    ├─ 有 request_prefix_state                     │
     │    │  且不是 must_answer_now？                 │
     │    │    ├─ 否 → continuation_messages = None      │
     │    │    └─ 是 → 调用 request_prefix_state.extend()│
     │    │         ├─ 上下文、工具、原有源消息发生变化   │
     │    │         │    └─ 返回 None，重新构建请求       │
     │    │         └─ 都符合复用条件                     │
     │    │              └─ 上次 sent_messages + 新消息  │
     │    │                                              │
     │    ├─ 计算 cache_prefix_reused                    │
     │    ├─ 计算 cache_prefix_message_count             │
     │    └─ 选择 prepare() 的输入                       │
     │         ├─ 复用：continuation_messages             │
     │         │        history_count=0                  │
     │         │        summary_state=None               │
     │         └─ 不复用：完整 request_messages           │
     │                    真实历史边界和摘要状态          │
     │                                                   │
     ├─ 7. context_manager.prepare()                     │
     │    ├─ 估算上下文 token                             │
     │    ├─ 必要时缩减工具输出                           │
     │    └─ 必要时生成历史摘要                           │
     │         │                                         │
     │         └─ 正在复用前缀，但需要首次历史摘要？       │
     │              └─ 满足回退条件时：                   │
     │                   清除前缀复用标记                 │
     │                   用原始请求及真实历史边界         │
     │                   再执行一次 prepare()            │
     │                                                   │
     ├─ 8. 更新摘要和缓存统计                             │
     │    ├─ 如果仍在复用前缀                             │
     │    │    ├─ 保留原来的摘要状态和覆盖位置             │
     │    │    └─ 检查 prepare 后的开头是否仍等于旧前缀  │
     │    │         └─ 不相同 → 取消复用标记，数量清零   │
     │    ├─ 把摘要覆盖位置换回原始历史坐标               │
     │    └─ 累计摘要的 usage、模型调用次数、预算 token   │
     │                                                   │
     ├─ 9. 第二次预算检查：摘要可能产生了新消耗           │
     │    ├─ exceeded=True → 结束 Run                    │
     │    ├─ should_warn=True，之前未加入警告             │
     │    │    ├─ 向请求追加警告                         │
     │    │    └─ warning_appended_after_prepare = True  │
     │    └─ should_finalize=True，之前尚未进入收尾       │
     │         ├─ 切换为收尾工具，或清空工具直接回答       │
     │         ├─ 必要时降低输出 token 上限              │
     │         └─ 追加收尾指令，发出收尾事件               │
     │                                                   │
     ├─ 10. 最后检查并保存前缀                            │
     │    ├─ 输入仍超过 input_budget？                   │
     │    │    └─ 是 → 返回上下文超限错误                 │
     │    ├─ 工具列表改变 → 取消旧前缀复用标记           │
     │    └─ 是否可以保存本轮前缀？                       │
     │         ├─ 非强制回答，且没有 prepare 后追加警告   │
     │         │    └─ 保存 RequestPrefixState           │
     │         └─ 否 → request_prefix_state = None       │
     │                                                   │
     ├─ 11. 请求模型 complete_stream()                   │
     │    ├─ 通过回调发出文本增量                         │
     │    ├─ 请求失败 → 返回模型错误                     │
     │    └─ 请求成功                                     │
     │         ├─ 累计模型用量                           │
     │         ├─ 保存 assistant_message                 │
     │         └─ 清除上一条修正提示                       │
     │                                                   │
     ├─ 12. 判断模型回复                                 │
     │    │                                              │
     │    ├─ 强制回答时仍发起工具调用                     │
     │    │    └─ 不执行，返回错误                       │
     │    │                                              │
     │    ├─ 没有工具调用                                 │
     │    │    ├─ 正文为空                               │
     │    │    │    ├─ 允许修正且未重试过                 │
     │    │    │    │    └─ 添加修正提示 → 下一 Step ────┤
     │    │    │    └─ 否 → 返回错误                     │
     │    │    ├─ 正文看起来是文本形式的工具调用           │
     │    │    │    ├─ 允许修正且未重试过                 │
     │    │    │    │    └─ 添加修正提示 → 下一 Step ────┤
     │    │    │    └─ 否 → 返回错误                     │
     │    │    └─ 正常正文 → 返回最终回答，结束 Run      │
     │    │                                              │
     │    └─ 有工具调用 → 继续执行工具                   │
     │                                                   │
     └─ 13. 执行本轮工具并更新状态                        │
          ├─ ToolRoundExecutor.execute()                 │
          ├─ 保存工具记录和工具结果消息                   │
          ├─ 更新 previous_signature、repeated_count     │
          ├─ 重复调用达到限制 → 返回错误                 │
          ├─ 保存 ToolRound                              │
          ├─ activated_tools.update(pending_activations) │
          ├─ 判断是否需要额外收尾步                       │
          │    └─ 更新 needs_extra_answer_step              │
          └─ 进入下一 Step ──────────────────────────────┘
'''
