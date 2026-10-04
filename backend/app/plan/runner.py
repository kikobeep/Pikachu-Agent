"""计划步骤的事件驱动推进器。

PlanRunner 不执行模型或工具调用，只负责在现有 AgentLoop 的边界上推进
PlanStep 状态。这样计划执行不会产生嵌套的 AgentLoop。
"""

from __future__ import annotations

from app.agent.emitter import EventEmitter
from app.agent.events import AgentEventType
from app.plan.config import Plan, PlanPatch, PlanStatus, PlanStepStatus
from app.plan.store import PlanStore


class PlanRunner:
    """在 Run 开始和工具轮结束时推进当前会话的计划。"""

    def __init__(self, store: PlanStore) -> None:
        self._store = store

    async def on_run_started(
        self,
        *,
        conversation_id: str | None,
        run_id: str | None = None,
        emitter: EventEmitter | None = None,
    ) -> Plan | None:
        """恢复计划，并确保存在可执行的当前步骤。"""

        if not conversation_id:
            return None
        plan = await self._store.plan_for_conversation(conversation_id)
        if plan is None:
            return None
        result = await self._reconcile(
            plan,
            conversation_id=conversation_id,
            run_id=run_id,
        )
        await self._emit_snapshot(result, emitter=emitter)
        return result

    async def on_plan_created(
        self,
        *,
        conversation_id: str | None,
        plan_id: str | None,
        run_id: str | None = None,
        emitter: EventEmitter | None = None,
    ) -> Plan | None:
        """在 plan_create 成功后激活新计划的第一个可执行步骤。"""

        if not conversation_id or not plan_id:
            return None
        plan = await self._store.resolve(
            plan_id,
            owner_conversation_id=conversation_id,
        )
        if plan is None:
            return None
        result = await self._reconcile(
            plan,
            conversation_id=conversation_id,
            run_id=run_id,
        )
        await self._emit_snapshot(result, emitter=emitter)
        return result

    async def on_tool_round_finished(
        self,
        *,
        conversation_id: str | None,
        run_id: str | None = None,
        plan_id: str | None = None,
        emitter: EventEmitter | None = None,
    ) -> Plan | None:
        """工具轮结束后检查是否可以激活下一个步骤。

        工具成功本身不会自动把步骤标记为 done；完成状态必须由模型通过
        plan_update 提供依据。Runner 只负责在状态已经明确后推进下一步。
        """

        if not conversation_id:
            return None
        plan = (
            await self._store.resolve(
                plan_id,
                owner_conversation_id=conversation_id,
            )
            if plan_id
            else await self._store.plan_for_conversation(conversation_id)
        )
        if plan is None:
            return None
        result = await self._reconcile(
            plan,
            conversation_id=conversation_id,
            run_id=run_id,
        )
        if result.revision != plan.revision:
            await self._emit_snapshot(result, emitter=emitter)
        return result

    @staticmethod
    async def _emit_snapshot(
        plan: Plan,
        *,
        emitter: EventEmitter | None,
    ) -> None:
        if emitter is None:
            return
        await emitter.emit(
            AgentEventType.PLAN_UPDATED,
            plan_id=plan.id,
            plan_title=plan.title,
            plan_status=plan.status.value,
            plan_revision=plan.revision,
            plan_steps=tuple(
                {
                    "id": step.id,
                    "title": step.title,
                    "status": step.status.value,
                    "note": step.note,
                }
                for step in plan.steps
            ),
        )

    async def _reconcile(
        self,
        plan: Plan,
        *,
        conversation_id: str,
        run_id: str | None,
    ) -> Plan:
        if plan.status in {
            PlanStatus.PAUSED,
            PlanStatus.COMPLETED,
            PlanStatus.FAILED,
            PlanStatus.CANCELLED,
        }:
            return plan

        if any(step.status is PlanStepStatus.BLOCKED for step in plan.steps):
            if plan.status is not PlanStatus.PAUSED:
                return await self._store.apply_patch(
                    plan.id,
                    PlanPatch(
                        status=PlanStatus.PAUSED,
                        expected_revision=plan.revision,
                        run_id=run_id,
                    ),
                    owner_conversation_id=conversation_id,
                )
            return plan

        current = next(
            (step for step in plan.steps if step.status is PlanStepStatus.IN_PROGRESS),
            None,
        )
        if current is not None:
            if plan.status is PlanStatus.PENDING:
                return await self._store.apply_patch(
                    plan.id,
                    PlanPatch(
                        status=PlanStatus.ACTIVE,
                        expected_revision=plan.revision,
                        run_id=run_id,
                    ),
                    owner_conversation_id=conversation_id,
                )
            return plan

        if plan.steps and all(
            step.status is PlanStepStatus.DONE for step in plan.steps
        ):
            if plan.status is not PlanStatus.COMPLETED:
                return await self._store.apply_patch(
                    plan.id,
                    PlanPatch(
                        status=PlanStatus.COMPLETED,
                        expected_revision=plan.revision,
                        run_id=run_id,
                    ),
                    owner_conversation_id=conversation_id,
                )
            return plan

        next_step = next(
            (step for step in plan.steps if step.status is PlanStepStatus.TODO),
            None,
        )
        if next_step is None:
            return plan

        return await self._store.apply_patch(
            plan.id,
            PlanPatch(
                status=PlanStatus.ACTIVE,
                step_id=next_step.id,
                step_status=PlanStepStatus.IN_PROGRESS,
                expected_revision=plan.revision,
                run_id=run_id,
            ),
            owner_conversation_id=conversation_id,
        )


__all__ = ["PlanRunner"]
