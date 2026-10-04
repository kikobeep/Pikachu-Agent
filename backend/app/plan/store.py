
from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.plan.config import Plan, PlanPatch, PlanStatus, PlanStep, PlanStepStatus

_CONTEXT_PLAN_STATUSES = frozenset(
    {PlanStatus.PENDING, PlanStatus.ACTIVE, PlanStatus.PAUSED}
)
DEFAULT_PLANS_DIR = Path(__file__).resolve().parents[2] / ".database" / "plans"

_TERMINAL_STATUSES = frozenset(
    {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED}
)


class PlanStore:
    def __init__(self, plans_dir: str | Path = DEFAULT_PLANS_DIR) -> None:
        self.plans_dir = Path(plans_dir).expanduser().resolve()
        self._locks: dict[str, asyncio.Lock] = {}


    async def initialize(self):
        await asyncio.to_thread(self.plans_dir.mkdir, parents=True, exist_ok=True)

    async def _all_plans(self) -> list[Plan]:
        """读取 plans 目录中所有持久化计划，按更新时间倒序。"""

        def _read() -> list[Plan]:
            plans: list[Plan] = []
            for path in sorted(self.plans_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                try:
                    plans.append(Plan.model_validate(payload))
                except ValueError:
                    continue
            plans.sort(key=lambda plan: plan.updated_at, reverse=True)
            return plans

        return await asyncio.to_thread(_read)


    async def plan_for_conversation(
        self,
        conversation_id: str,
    ) -> Plan | None:


        normalized = _normalize(
            conversation_id,
            field_name="conversation_id",
        )
        plans = [
            plan
            for plan in await self._all_plans()
            if normalized == plan.owner_conversation_id
            and plan.status in _CONTEXT_PLAN_STATUSES
        ]
        plans.sort(key=lambda plan: plan.updated_at, reverse=True)
        return plans[0] if plans else None

    async def create(
        self,
        *,
        title: str,
        description: str | None,
        goal: str | None,
        priority,
        steps,
        owner_conversation_id: str,
        run_ids,
    ) -> Plan:
        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        now = datetime.now(UTC)
        plan = Plan(
            id=uuid4().hex,
            title=title,
            description=description,
            goal=goal,
            priority=priority,
            steps=tuple(steps),
            owner_conversation_id=normalized_owner,
            run_ids=tuple(run_ids),
            created_at=now,
            updated_at=now,
        )
        await asyncio.to_thread(
            _write_plan,
            self.plans_dir / f"{plan.id}.json",
            plan,
        )
        return plan

    async def apply_patch(
        self,
        plan_id: str,
        patch: PlanPatch,
        *,
        owner_conversation_id: str,
    ) -> Plan:
        plan = await self.resolve(
            plan_id,
            owner_conversation_id=owner_conversation_id,
        )
        if plan is None:
            raise KeyError(f"计划不存在：{plan_id}")

        if (
            patch.expected_revision is not None
            and patch.expected_revision != plan.revision
        ):
            raise ValueError(
                f"任务版本冲突：期望 {patch.expected_revision}，"
                f"当前 {plan.revision}"
            )

        data = plan.model_dump()
        fields = patch.model_fields_set

        if "goal" in fields:
            data["goal"] = patch.goal
        if patch.status is not None:
            data["status"] = patch.status
        if "state" in fields:
            data["state"] = patch.state or ()
        if patch.add_constraints:
            data["constraints"] = (*plan.constraints, *patch.add_constraints)
        if patch.add_key_facts:
            data["key_facts"] = (*plan.key_facts, *patch.add_key_facts)
        if patch.step_id is not None:
            data["steps"] = _update_step(
                plan.steps,
                patch.step_id,
                patch.step_status,
                patch.step_note,
            )
        if patch.run_id is not None and patch.run_id not in plan.run_ids:
            data["run_ids"] = (*plan.run_ids, patch.run_id)

        data["updated_at"] = datetime.now(UTC)
        data["revision"] = plan.revision + 1
        if patch.status in _TERMINAL_STATUSES:
            data["completed_at"] = data["updated_at"]
        elif patch.status is not None:
            data["completed_at"] = None

        updated = Plan.model_validate(data)
        await asyncio.to_thread(
            _write_plan,
            self.plans_dir / f"{updated.id}.json",
            updated,
        )
        return updated

    async def list(
        self,
        *,
        limit: int,
        status: PlanStatus | None,
        owner_conversation_id: str,
    ) -> list[Plan]:
        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        plans = [
            plan
            for plan in await self._all_plans()
            if plan.owner_conversation_id == normalized_owner
        ]
        if status is not None:
            plans = [plan for plan in plans if plan.status is status]
        return plans[:limit]

    async def resolve(
        self,
        plan_id: str,
        *,
        owner_conversation_id: str,
    ) -> Plan | None:
        """按完整 ID 或唯一前缀解析计划；模糊时抛错，不存在返回 None。"""

        normalized_owner = _normalize(
            owner_conversation_id,
            field_name="owner_conversation_id",
        )
        prefix = plan_id.strip().lower()
        if not prefix:
            return None
        matches = [
            plan
            for plan in await self._all_plans()
            if plan.owner_conversation_id == normalized_owner
            and plan.id.startswith(prefix)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"计划 ID 前缀不唯一：{plan_id}")
        return matches[0]

    async def active_plan_for_conversation(
        self,
        conversation_id: str,
    ) -> Plan | None:
        return await self.plan_for_conversation(conversation_id)

def _update_step(
    steps: tuple[PlanStep, ...],
    step_id: str,
    step_status: PlanStepStatus | None,
    step_note: str | None,
) -> tuple[PlanStep, ...]:
    """就地更新单个步骤的状态与备注。"""

    updated: list[PlanStep] = []
    found = False
    for step in steps:
        if step.id == step_id:
            found = True
            data = step.model_dump()
            if step_status is not None:
                data["status"] = step_status
            if step_note is not None:
                data["note"] = step_note
            updated.append(PlanStep.model_validate(data))
        else:
            updated.append(step)
    if not found:
        raise ValueError(f"步骤不存在：{step_id}")
    return tuple(updated)


def _write_plan(path: Path, plan: Plan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized
