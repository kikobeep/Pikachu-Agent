
from __future__ import annotations

from dataclasses import dataclass

from app.model.config import Message
from app.agent.events import AgentEventType
from app.agent.utils import skill_read_outcome
from app.memory.manager import MemoryManager
from app.memory.search_config import MemorySearchInputs, SearchMode,MemorySearchResult
from app.skills.context import SkillContextProvider
from app.skills.store import SkillStore
from app.plan.context import PlanContextProvider
from app.tools.config import ToolResult
from app.skills.config import Skill
from app.agent.emitter import EventEmitter
from app.checkpoint import render_checkpoint_context,RunCheckpoint


@dataclass(frozen=True, slots=True)
class ContextInjection:
    """本次请求要补充的上下文消息，以及技能和记忆检索信息。"""

    messages: tuple[Message, ...]

    available_skill_count: int | None
    skill_meta_token_count: int | None

    activated_skill_names: tuple[str, ...]
    activated_skill_token_count: int | None
    activated_skill_message_names: tuple[str, ...]

    memory_candidate_ids: tuple[str, ...] = ()
    memory_search_mode: str | None = None



class ContextRuntime:
    def __init__(
        self,
        *,
        memory_manager: MemoryManager | None = None,
        memory_search_query: MemorySearchInputs | None = None,
        skill_context_provider: SkillContextProvider | None,
        skill_store: SkillStore | None,
        plan_context_provider: PlanContextProvider | None
    ):
        # memory
        self._memory_manager = memory_manager
        self. _memory_loaded = False
        self._memory_messages: tuple[Message, ...] = ()
        self._memory_search_input = memory_search_query
        self._memory_search_result: MemorySearchResult | None = None

        # skill
        self._skill_context_provider = skill_context_provider
        self._skill_store = skill_store
        self._skill_loaded = False
        self._active_skills: dict[str, Skill] = {}

        # plan
        self._plan_context_provider = plan_context_provider

    @property
    def active_skill_names(self) -> tuple[str, ...]:
        return tuple(self._active_skills)

    async def build(
        self,
        *,
        conversation_id: str | None,
        recovery_checkpoint: RunCheckpoint | None
    ):
        '''
        缓存返回的消息，所以同一个 Run 的后续 Step 会复用这份记忆上下文，不再重复检索
        '''
        messages: list[Message] = []
        # 
        if self._memory_manager is not None and not self._memory_loaded:
            self._memory_loaded = True
            try:
                self._memory_messages = await self._load_memory_messages(conversation_id)
            except Exception:
                self._memory_messages = ()
        messages.extend(self._memory_messages)

        # skill
        if self._skill_context_provider and self._skill_store:
            if not self._skill_loaded:
                self._skill_loaded = True
            try:
                self._skill_metadata = await self._skill_store.list_metadata()
            except Exception:
                self._skill_metadata = ()
            metadata_message = self._skill_context_provider.get_meta_message(self._skill_metadata)
            if metadata_message is not None:
                messages.append(metadata_message)
        
        # active skill
        injected_active_names: tuple[str, ...] = ()
        if self._skill_context_provider is not None and self._active_skills:
            active_messages = self._skill_context_provider.active_messages(
                tuple(self._active_skills.values())
            )
            if active_messages:
                injected_active_names = tuple(self._active_skills)
                messages.extend(active_messages)
        
        # checkpoint recovery，上一次运行中断时保存的状态来帮助 Agent 恢复任务
        if recovery_checkpoint is not None:
            messages.append(render_checkpoint_context(recovery_checkpoint))
        

        if self._plan_context_provider is not None:
            plan = await self._plan_context_provider.load_messages(conversation_id)
            messages.append(plan)



        return ContextInjection(
            messages = messages,
            available_skill_count = len(self._skill_metadata) if self._skill_context_provider else None,
            skill_meta_token_count = self._skill_context_provider.meta_tokens(self._skill_metadata) if self._skill_context_provider else None,
            activated_skill_names = tuple(self._active_skills),
            activated_skill_token_count = self._skill_context_provider.active_tokens(tuple(self._active_skills.values())) if self._skill_context_provider else None,
            activated_skill_message_names = injected_active_names,
            memory_candidate_ids=tuple(
                candidate.memory_id
                for candidate in (
                    self._memory_search_result.candidates
                    if self._memory_search_result is not None
                    else ()
                )
            ),
            memory_search_mode=(
                self._memory_search_result.mode.value
                if self._memory_search_result is not None
                else None
            )
        )


    async def _emit_activation_failed(
        self,
        emitter: EventEmitter,
        *,
        step: int,
        skill_name: str,
        error: str,
    ) -> None:
        await emitter.emit(
            AgentEventType.SKILL_ACTIVATION_FAILED,
            step=step,
            skill_name=skill_name,
            skill_error=error,
        )

    async def activate_skill(
        self,
        result: ToolResult,
        *,
        emitter: EventEmitter,
        step: int
    ):
        provider = self._skill_context_provider
        if self._skill_store is None or provider is None:
            return

        skill_name, found = skill_read_outcome(result.output)
        if not skill_name:
            return
        if not found:
            await self._emit_activation_failed(
                emitter,
                step=step,
                skill_name=skill_name,
                error="skill not found",
            )
            return
        if skill_name in self._active_skills:
            return
        skill = await self._skill_store.load(skill_name)
        if skill is None:
            await emitter.emit(
                AgentEventType.SKILL_ACTIVATION_FAILED,
                step=step,
                skill_name=skill_name,
                skill_error="skill not found",
            )
            return
        
        if provider.check_budget(
            tuple(self._active_skills.values()),
            skill,
        ):
            await emitter.emit(
                AgentEventType.SKILL_ACTIVATION_FAILED,
                step=step,
                skill_name=skill_name,
                skill_error="active skill context budget exceeded",
            )
            return
        self._active_skills[skill.metadata.name] = skill
        await emitter.emit(
            AgentEventType.SKILL_ACTIVATED,
            step=step,
            skill_name=skill.metadata.name,
            skill_scope=skill.metadata.scope.value,
            active_skill_names=tuple(self._active_skills),
            active_skill_tokens=provider.active_tokens(
                tuple(self._active_skills.values())
            ),
        )


    async def _load_memory_messages(
        self,
        conversation_id: str | None,
    ) -> tuple[Message, ...]:
        manager = self._memory_manager
        if manager is None:
            return ()
        hybrid = manager.hybrid_search_enabled
        if not hybrid or self._memory_search_query is None:
            return await manager.memory_context()
        try:
            query = await self._memory_search_with_task(conversation_id = conversation_id)
            result = await manager.search(query)
        except Exception:
            return await manager.memory_context()
        self._memory_search_result = result
        if result.mode is SearchMode.UNAVAILABLE:
            return await manager.memory_context(result)
        return await manager.memory_context(search_result=result)



    
    async def _memory_search_with_task(
        self,
        conversation_id: str | None,
    ) -> MemorySearchInputs:

        plan_title: str | None = None
        plan_steps: tuple[str, ...] = ()

        if self._plan_context_provider is not None:
            plan_title, plan_steps = (
                await self._plan_context_provider.plan_for_memory(
                    conversation_id
                )
            ) 
        return self._memory_search_input.with_task(plan_title, plan_steps)
