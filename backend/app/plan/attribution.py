"""把当前 Plan 工作位置适配为中立工具输出归因。"""

from __future__ import annotations

from app.plan.config import PlanStepStatus
from app.tools.output import ToolOutputAttribution

from .store import PlanStore


class PlanToolOutputAttributionResolver:
    """读取当前活动 Plan 与唯一执行中 Step，不依赖 Evidence 领域。"""

    def __init__(self, store: PlanStore) -> None:
        self._store = store

    async def resolve(self, conversation_id: str) -> ToolOutputAttribution:
        plan = await self._store.active_plan_for_conversation(conversation_id)
        if plan is None:
            return ToolOutputAttribution()
        step = next(
            (
                item
                for item in plan.steps
                if item.status is PlanStepStatus.IN_PROGRESS
            ),
            None,
        )
        return ToolOutputAttribution(
            plan_id=plan.id,
            plan_step_id=step.id if step is not None else None,
        )


__all__ = ["PlanToolOutputAttributionResolver"]
