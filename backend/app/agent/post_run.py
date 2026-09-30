from __future__ import annotations

from typing import Any, Protocol

from app.agent.events import AgentEventType
from app.agent.result import AgentResult, AgentStopReason
from app.memory.archive_config import ArchiveAction, MemoryArchiveInput
from app.memory.archive_model import ArchiveMemoryReflector
from app.memory.manager import MemoryManager
from app.memory.reflection_config import MemoryReflectionInput, ReflectionAction
from app.memory.reflection_gate import decide_reflection_gate
from app.memory.reflection_model import MemoryReflector
from app.model.config import ModelUsage

from .utils import get_memory_versions


class EventEmitter(Protocol):
    async def emit(self, event: AgentEventType, **data: Any) -> None: ...


class MemoryCoordinator:
    """
    判断是否要更新（update) 或者 新增（add）记忆
    """

    def __init__(
        self,
        manager: MemoryManager,
        reflector: MemoryReflector,
        archive_reflector: ArchiveMemoryReflector,
    ):
        self._reflector = reflector
        self._manager = manager
        self._archive_reflector = archive_reflector

    async def planner(
        self,
        result: AgentResult,
        *,
        user_input: str,
        conversation_id: str | None,
        emitter: EventEmitter,
    ):
        if not self._reflector:  # 如果没有记忆反思模型，直接handler post
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason="memory reflection model is None",
            )
            return
        if not self._reflector.enabled:  # 如果记忆模型的开关没有打开，直接handler post
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason="memory reflection is disabled",
            )
            return
        if (
            result.stop_reason is not AgentStopReason.FINAL_ANSWER
        ):  # agent对话不是正常结束
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason=f"stop_reason={result.stop_reason.value}",
            )
            return

        memory_versions = get_memory_versions(result.tool_calls)

        gate = decide_reflection_gate(
            user_input=user_input, memory_ids=tuple(memory_versions)
        )

        # 如果不需要调用反思模型
        if not gate.should_reflect:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_SKIPPED,
                reflection_triggered=False,
                reflection_skip_reason=f"gate:{gate.reason.value}",
            )
            return
        try:
            # 读取已有的核心记忆和普通记忆目录
            core_memory, regular_memory_index = await self._manager.reflection_context()
            reflection_input = MemoryReflectionInput(
                run_id=result.run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                final_answer=result.final_message.content or "",
                tool_context=tuple(
                    record.result.model_dump_json() for record in result.tool_calls
                ),
                memory_ids=tuple(memory_versions),
                core_memory=core_memory,
                memory_index=regular_memory_index,
            )
        except Exception as exc:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_FAILED,
                reflection_triggered=True,
                reflection_error=f"{type(exc).__name__}: {exc}",
                reflection_mutation_applied=False,
                provider=self._reflector.provider_hint,
                model=self._reflector.model_hint,
                usage=ModelUsage(),
            )
            if self._manager is not None:
                await self._ensure_capacity(required_slots=0, emitter=emitter)
            return

        try:
            await self._run_reflection(
                reflection_input=reflection_input,
                memory_revisions=memory_versions,
                emitter=emitter,
            )
        except Exception as exc:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_FAILED,
                reflection_triggered=True,
                reflection_error=f"{type(exc).__name__}: {exc}",
                reflection_mutation_applied=False,
                provider=self._reflector.provider_hint,
                model=self._reflector.model_hint,
                usage=ModelUsage(),
            )

    async def _run_reflection(
        self,
        reflection_input: MemoryReflectionInput,
        memory_revisions: dict[str, int],
        emitter: EventEmitter,
    ) -> None:
        await emitter.emit(
            AgentEventType.MEMORY_REFLECTION_STARTED,
            reflection_triggered=True,
            provider=self._reflector.provider_hint,
            model=self._reflector.model_hint,
        )

        response = await self._reflector.decide(reflection_input=reflection_input)
        decision = response.decision
        if response.error is not None:
            raise ValueError(response.error)
        if decision is None:
            await emitter.emit(
                AgentEventType.MEMORY_REFLECTION_COMPLETED,
                reflection_triggered=True,
                reflection_action="none",
                provider=response.provider,
                model=response.model,
                usage=response.usage,
            )
            return

        # 对已有记忆进行修改
        if decision.action is ReflectionAction.UPDATE:
            memory_id = decision.memory_id
            expected_revisions = memory_revisions.get(memory_id)
            if expected_revisions is None:
                raise ValueError(
                    "reflection update requires memory_read success in current run"
                )
            # 修改当前的memery
            await self._manager.update_regular_memory(
                memory_id or "",
                expected_revision=expected_revisions,
                title=decision.title or "",
                summary=decision.summary or "",
                content=decision.content or "",
                reason=decision.reason,
            )

        # 对已有记忆进行新增
        elif decision.action is ReflectionAction.CREATE:
            # 判断是否有足够的空间可以新增
            capacity_ready = await self._ensure_capacity(
                required_slots=1,
                emitter=emitter,
            )
            if not capacity_ready:
                raise RuntimeError(
                    "memory create skipped because capacity is unavailable"
                )

            # 如果有足够的空间，则新增记忆
            record = await self._manager.create_if_capacity(
                title=decision.title or "",
                summary=decision.summary or "",
                content=decision.content or "",
            )
            if record is None:
                raise RuntimeError(
                    "memory create lost the available slot to a concurrent run"
                )

            # 新增记忆之后，再对现有的记忆容量进行维护

            if await self._manager.is_over_capacity():
                await self._ensure_capacity(required_slots=0, emitter=emitter)

        await emitter.emit(
            AgentEventType.MEMORY_REFLECTION_COMPLETED,
            reflection_triggered=True,
            reflection_action=decision.action.value,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
        )

    async def _ensure_capacity(
        self, *, required_slots: int, emitter: EventEmitter
    ) -> bool:
        if self._manager is None:
            return False
        active_count = await self._manager.active_count()
        if active_count + required_slots <= self._manager.max_active:
            return True

        
        if self._archive_reflector is None or not self._archive_reflector.enabled:
            await emitter.emit(
                AgentEventType.MEMORY_ARCHIVE_SKIPPED,
                archive_triggered=False,
                archive_skip_reason=(
                    "unavailable" if self._archive_reflector is None else "disabled"
                ),
                archive_active_count=active_count,
                archive_max_active=self._manager.max_active,
                archive_remaining_overflow=max(
                    0,
                    active_count + required_slots - self._manager.max_active,
                ),
            )
            return False
        # 如果记忆空间不够，选择一批旧记忆作为候选，把旧记忆做归档，为新记忆腾出active空间
        for _ in range(self._archive_reflector.config.max_actions):
            candidates = await self._manager.get_candidates(
                limit=self._archive_reflector.config.candidate_limit
            )
            candidate_ids = tuple(candidate.id for candidate in candidates)
            remaining = active_count + required_slots - self._manager.max_active
            if not candidates:
                await emitter.emit(
                    AgentEventType.MEMORY_ARCHIVE_FAILED,
                    archive_triggered=True,
                    archive_error="no active archive candidates",
                    archive_active_count=active_count,
                    archive_max_active=self._manager.max_active,
                    archive_remaining_overflow=max(0, remaining),
                )
                return False
            await emitter.emit(
                AgentEventType.MEMORY_ARCHIVE_STARTED,
                archive_triggered=True,
                archive_active_count=active_count,
                archive_max_active=self._manager.max_active,
                archive_candidate_ids=candidate_ids,
                archive_remaining_overflow=max(0, remaining),
                provider=self._archive_reflector.provider_hint,
                model=self._archive_reflector.model_hint,
            )
            response = await self._archive_reflector.decide(
                MemoryArchiveInput(
                    active_count=active_count,
                    max_active=self._manager.max_active,
                    required_slots=required_slots,
                    candidates=candidates,
                )
            )
            if response.error is not None or response.decision is None:
                await emitter.emit(
                    AgentEventType.MEMORY_ARCHIVE_FAILED,
                    archive_triggered=True,
                    archive_duration_ms=response.duration_ms,
                    archive_error=(
                        response.error or "archive model returned no decision"
                    ),
                    archive_active_count=active_count,
                    archive_max_active=self._manager.max_active,
                    archive_candidate_ids=candidate_ids,
                    archive_remaining_overflow=max(0, remaining),
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                )
                return False

            decision = response.decision
            if decision.action is ArchiveAction.DEFER:
                await emitter.emit(
                    AgentEventType.MEMORY_ARCHIVE_COMPLETED,
                    archive_triggered=True,
                    archive_action=decision.action.value,
                    archive_duration_ms=response.duration_ms,
                    archive_reason=decision.reason,
                    archive_active_count=active_count,
                    archive_max_active=self._manager.max_active,
                    archive_candidate_ids=candidate_ids,
                    archive_remaining_overflow=max(0, remaining),
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                )
                return False

            memory_id = decision.memory_id or ""
            snapshots = {candidate.id: candidate for candidate in candidates}
            if snapshots.get(memory_id) is None:
                await emitter.emit(
                    AgentEventType.MEMORY_ARCHIVE_FAILED,
                    archive_triggered=True,
                    archive_action=decision.action.value,
                    archive_duration_ms=response.duration_ms,
                    archive_error=(
                        "archive selected an ID outside the candidate set"
                    ),
                    archive_memory_id=memory_id,
                    archive_active_count=active_count,
                    archive_max_active=self._manager.max_active,
                    archive_candidate_ids=candidate_ids,
                    archive_remaining_overflow=max(0, remaining),
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                )
                return False

            expected_record = snapshots.get(memory_id)
            try:
                archived = await self._manager.archive_if_unchanged(
                    memory_id, expected_record=expected_record, reason=decision.reason
                )

            except Exception as exc:
                await emitter.emit(
                    AgentEventType.MEMORY_ARCHIVE_FAILED,
                    archive_triggered=True,
                    archive_action=decision.action.value,
                    archive_duration_ms=response.duration_ms,
                    archive_error=f"{type(exc).__name__}: {exc}",
                    archive_memory_id=memory_id,
                    archive_reason=decision.reason,
                    archive_active_count=active_count,
                    archive_max_active=self._manager.max_active,
                    archive_candidate_ids=candidate_ids,
                    archive_remaining_overflow=max(0, remaining),
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                )
                return False

            after_count = await self._manager.active_count()
            after_remaining = max(
                0,
                after_count + required_slots - self._manager.max_active,
            )
            await emitter.emit(
                AgentEventType.MEMORY_ARCHIVE_COMPLETED,
                archive_triggered=True,
                archive_action=decision.action.value,
                archive_duration_ms=response.duration_ms,
                archive_memory_id=archived.id,
                archive_reason=decision.reason,
                archive_active_count=after_count,
                archive_max_active=self._manager.max_active,
                archive_candidate_ids=candidate_ids,
                archive_remaining_overflow=after_remaining,
                provider=response.provider,
                model=response.model,
                usage=response.usage,
            )

        active_count = await self._manager.active_count()
        remaining = max(0, active_count + required_slots - self._manager.max_active)
        if remaining == 0:
            return True
        await emitter.emit(
            AgentEventType.MEMORY_ARCHIVE_SKIPPED,
            archive_triggered=True,
            archive_skip_reason="max_actions_reached",
            archive_active_count=active_count,
            archive_max_active=self._manager.max_active,
            archive_remaining_overflow=remaining,
        )
        return False
