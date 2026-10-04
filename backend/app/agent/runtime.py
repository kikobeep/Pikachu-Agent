
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from contextlib import suppress
from uuid import uuid4

from app.agent.budget import RunBudget, RunBudgetConfig
from app.model.config import AgentMode, Message, MessageRole
from app.agent.events import AgentEventHandler, AgentEventType, NullEventHandler
from app.agent.loop import AgentLoop
from app.agent.result import AgentResult
from app.agent.post_run import MemoryCoordinator
from app.checkpoint.config import RunCheckpoint
from app.checkpoint.store import CheckpointStore
from app.context.handoff_store import HandoffStore
from app.context.manager import ContextManager
from app.context.summary import ConversationSummaryState
from app.memory.archive_model import ArchiveMemoryReflector
from app.memory.manager import MemoryManager
from app.memory.reflection_model import MemoryReflector
from app.model.config import ModelProvider
from app.model.registry import ModelAdapterRegistry
from app.skills.context import SkillContextProvider
from app.skills.store import SkillStore
from app.plan.context import PlanContextProvider
from app.plan.runner import PlanRunner
from app.tools.approval import ApprovalGate
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolHook
from app.tools.output import ToolOutputRecorder
from app.tools.permissions.policy import PermissionPolicyEngine
from app.tools.permissions.store import PermissionRuleStore
from app.tools.register import ToolRegistry
from app.agent.emitter import EventEmitter
from app.ace import AceCoordinator, AceSelection

class AgentRuntime:
    """运行模型，直到返回最终消息或循环必须停止。"""

    def __init__(
        self,
        model_registry: ModelAdapterRegistry,
        tool_registry: ToolRegistry,
        *,
        provider: ModelProvider | str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        max_steps: int = 12,
        max_tool_rounds: int | None = None,
        max_output_tokens: int | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_hooks: Sequence[ToolHook] = (),
        approval_gate: ApprovalGate | None = None,
        policy_engine: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        context_manager: ContextManager | None = None,
        plan_context_provider: PlanContextProvider | None = None,
        plan_runner: PlanRunner | None = None,
        checkpoint_store: CheckpointStore | None = None,
        memory_manager: MemoryManager | None = None,
        memory_reflector: MemoryReflector | None = None,
        memory_archive_reflector: ArchiveMemoryReflector | None = None,
        skill_store: SkillStore | None = None,
        skill_context_provider: SkillContextProvider | None = None,
        tool_output_recorder: ToolOutputRecorder | None = None,
        run_budget_config: RunBudgetConfig | None = None,
        handoff_store: HandoffStore | None = None,
        workspace_root: str | Path | None = None,
        memory_auto_search_enabled: bool = True,
        ace: AceCoordinator | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if max_tool_rounds is not None and max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if memory_reflector is not None and memory_manager is None:
            raise ValueError("memory_reflector requires memory_manager")
        if memory_archive_reflector is not None and memory_manager is None:
            raise ValueError("memory_archive_reflector requires memory_manager")
        if skill_context_provider is not None and skill_store is None:
            raise ValueError("skill_context_provider requires skill_store")

        self._system_prompt = (
            system_prompt
            if system_prompt is not None and system_prompt.strip()
            else None
        )
        self._context_manager = context_manager or ContextManager()
        self._checkpoint_store = checkpoint_store
        self._ace = ace
        self._background_tasks: set[asyncio.Task[None]] = set()

        self._post_run = MemoryCoordinator(
            manager=memory_manager,
            reflector=memory_reflector,
            archive_reflector=memory_archive_reflector
        )
        self._run_budget = RunBudget(run_budget_config)
        self._tool_executor = tool_executor or ToolExecutor(
            tool_registry,
            approval_gate=approval_gate,
            policy_engine=policy_engine,
            rule_store=rule_store,
            hooks=tool_hooks,
            output_recorder=tool_output_recorder,
        )
        self._loop = AgentLoop(
            model_registry=model_registry,
            tool_registry=tool_registry,
            tool_executor=self._tool_executor,
            provider=provider,
            model=model,
            system_prompt=self._system_prompt,
            max_steps=max_steps,
            max_tool_rounds=max_tool_rounds,
            max_output_tokens=max_output_tokens,
            context_manager=self._context_manager,
            plan_context_provider=plan_context_provider,
            plan_runner=plan_runner,
            checkpoint_store=checkpoint_store,
            memory_manager=memory_manager,
            memory_auto_search_enabled=memory_auto_search_enabled,
            skill_store=skill_store,
            skill_context_provider=skill_context_provider,
            run_budget=self._run_budget,
            handoff_store=handoff_store,
            workspace_root=workspace_root,
        )

    def set_max_output_tokens(self, value: int | None) -> None:
        """Update the output cap for subsequent runs in this runtime."""
        if value is not None and value < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self._loop.set_max_output_tokens(value)

    async def flush_background_tasks(self) -> None:
        """Wait for detached ACE reflection tasks before application shutdown."""
        tasks = tuple(self._background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(
        self,
        user_input: str,
        *,
        history: Sequence[Message] = (),
        conversation_id: str | None = None,
        event_handler: AgentEventHandler | None = None,
        summary_state: ConversationSummaryState | None = None,
        run_id: str | None = None,
        recovery_run_id: str | None = None,
        mode: AgentMode = AgentMode.DEFAULT,
        source: str | None = None,
    ) -> AgentResult:

        run_id = run_id or uuid4().hex
        emitter = EventEmitter(
            handler=event_handler,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        ace_selection = (
            self._ace.select(user_input) if self._ace is not None else AceSelection()
        )
        try:
            recovery_checkpoint: RunCheckpoint | None = None
            if self._checkpoint_store is not None:
                if recovery_run_id is not None:
                    recovery_checkpoint = (
                        await self._checkpoint_store.get_unrecovered(
                            recovery_run_id
                        )
                    )
                await self._checkpoint_store.start(
                    run_id,
                    conversation_id=conversation_id,
                    user_message=Message(
                        role=MessageRole.USER,
                        content=user_input,
                    ),
                )
            try:
                result = await self._loop.run(
                    run_id,
                    user_input,
                    history=history,
                    conversation_id=conversation_id,
                    emitter=emitter,
                    summary_state=summary_state,
                    recovery_checkpoint=recovery_checkpoint,
                    mode=mode,
                    additional_system_prompt=ace_selection.prompt() or None,
                )
                if self._ace is not None:
                    ace_selection = self._ace.resolve_selection(
                        ace_selection,
                        result.content,
                    )
                    cleaned = self._ace.clean_generator_output(result.content)
                    if cleaned != result.content:
                        result = result.model_copy(
                            update={
                                "final_message": result.final_message.model_copy(
                                    update={"content": cleaned}
                                ),
                            }
                        )
            except BaseException as exc:
                if self._checkpoint_store is not None:
                    with suppress(Exception):
                        await self._checkpoint_store.interrupt(
                            run_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                raise

            if self._checkpoint_store is not None:
                if result.ok:
                    await self._checkpoint_store.complete(
                        run_id,
                        stop_reason=result.stop_reason,
                    )
                    if recovery_checkpoint is not None:
                        with suppress(Exception):
                            await self._checkpoint_store.mark_recovered(
                                recovery_checkpoint.run_id,
                                recovered_by_run_id=run_id,
                            )
                else:
                    await self._checkpoint_store.fail(
                        run_id,
                        stop_reason=result.stop_reason,
                        error=(
                            result.error.message
                            if result.error is not None
                            else None
                        ),
                    )
        
            await emitter.emit(
                (
                    AgentEventType.AGENT_COMPLETED
                    if result.ok
                    else AgentEventType.AGENT_FAILED
                ),
                step=result.steps or None,
                message=result.final_message,
                usage=result.usage,
                stop_reason=result.stop_reason,
                error=result.error,
                result=result,
            )
            await self._post_run.planner(
                result,
                user_input=user_input,
                conversation_id=conversation_id,
                emitter=emitter,
            )
            # Polyglot evaluation supplies the authoritative test result after
            # this method returns. Do not learn from an unlabelled initial run
            # or from the feedback repair run; the evaluator explicitly calls
            # AceCoordinator.reflect() for the initial trajectory.
            if self._ace is not None and source not in {"eval_initial", "eval_feedback"}:
                # ACE reflection is deliberately detached from the user-facing run:
                # a slow/failing reflector must never delay or change the answer.
                task = asyncio.create_task(
                    self._ace.reflect(
                        user_input=user_input,
                        result=result,
                        selection=ace_selection,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return result
        finally:
            with suppress(Exception):
                await self._tool_executor.clear_run_rules(run_id) # run_id 关联的临时授权规则。例如用户批准：本次运行中允许执行这类操作。Run 结束后，就应清除这份临时授权，避免延续到下一次运行


            
