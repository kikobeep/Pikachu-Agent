"""把当前 Task 工作位置适配为中立工具输出归因。"""

from __future__ import annotations

from app.task.config import TaskStepStatus
from app.tools.output import ToolOutputAttribution

from .store import PlanStore


class TaskToolOutputAttributionResolver:
    """读取当前活动 Task 与唯一执行中 Step，不依赖 Evidence 领域。"""

    def __init__(self, store: PlanStore) -> None:
        self._store = store

    async def resolve(self, conversation_id: str) -> ToolOutputAttribution:
        task = await self._store.active_plan_for_conversation(conversation_id)
        if task is None:
            return ToolOutputAttribution()
        step = next(
            (
                item
                for item in task.steps
                if item.status is TaskStepStatus.IN_PROGRESS
            ),
            None,
        )
        return ToolOutputAttribution(
            task_id=task.id,
            task_step_id=step.id if step is not None else None,
        )


__all__ = ["TaskToolOutputAttributionResolver"]
