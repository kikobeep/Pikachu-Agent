"""任务边界分类、任务 hash 与稳定窗口确认。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Sequence

from app.model.config import Message, MessageRole, ModelRequest, ModelUsage
from .summary import ConversationSummaryState, RollingConversationSummary


class TaskBoundaryDecision(StrEnum):
    SAME = "same"
    NEW = "new"
    UNCERTAIN = "uncertain"


class TaskBoundaryTrigger(StrEnum):
    AUTO = "AUTO"
    TASK = "TASK"


@dataclass(frozen=True)
class TaskBoundaryObservation:
    decision: TaskBoundaryDecision
    basis_message_id: str
    task_hash: str | None
    confirmed_change: bool
    stable_count: int
    required_stable_count: int


def basis_message_id(*, conversation_id: str | None, history: Sequence[Message], user_input: str) -> str:
    payload = json.dumps(
        {"conversation_id": conversation_id or "", "history_count": len(history), "user_input": user_input},
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    return "msg_" + hashlib.sha256(payload).hexdigest()[:24]


def task_hash(*, conversation_id: str | None, basis: str) -> str:
    payload = f"{conversation_id or ''}:{basis}".encode()
    return "task_" + hashlib.sha256(payload).hexdigest()[:16]


def ensure_initial_task(
    state: ConversationSummaryState | None,
    *,
    conversation_id: str | None,
    basis: str,
) -> ConversationSummaryState:
    if state is not None and state.active_task_hash:
        return state
    base = state or ConversationSummaryState(
        summary=RollingConversationSummary(), covered_message_count=0
    )
    return base.model_copy(update={"active_task_hash": task_hash(conversation_id=conversation_id, basis=basis)})


def observe(
    state: ConversationSummaryState,
    *,
    decision: TaskBoundaryDecision,
    conversation_id: str | None,
    basis: str,
    required_stable_count: int = 2,
) -> tuple[ConversationSummaryState, TaskBoundaryObservation]:
    if required_stable_count < 1:
        raise ValueError("required_stable_count must be at least 1")

    if decision is TaskBoundaryDecision.UNCERTAIN:
        updated = state.model_copy(update={
            "candidate_task_hash": None,
            "candidate_task_basis_message_id": None,
            "task_hash_stable_count": 0,
        })
        return updated, TaskBoundaryObservation(decision, basis, None, False, 0, required_stable_count)

    if decision is TaskBoundaryDecision.SAME:
        if not state.candidate_task_hash:
            updated = state.model_copy(update={
                "candidate_task_hash": None,
                "candidate_task_basis_message_id": None,
                "task_hash_stable_count": 0,
            })
            return updated, TaskBoundaryObservation(decision, basis, state.active_task_hash, False, 0, required_stable_count)
        candidate = state.candidate_task_hash
    else:
        candidate = task_hash(conversation_id=conversation_id, basis=basis)

    if candidate == state.active_task_hash:
        updated = state.model_copy(update={
            "candidate_task_hash": None,
            "candidate_task_basis_message_id": None,
            "task_hash_stable_count": 0,
        })
        return updated, TaskBoundaryObservation(decision, basis, candidate, False, 0, required_stable_count)

    count = state.task_hash_stable_count + 1 if candidate == state.candidate_task_hash else 1
    confirmed = count >= required_stable_count
    if confirmed:
        updated = state.model_copy(update={
            "active_task_hash": candidate,
            "candidate_task_hash": None,
            "candidate_task_basis_message_id": None,
            "task_hash_stable_count": 0,
        })
    else:
        updated = state.model_copy(update={
            "candidate_task_hash": candidate,
            "candidate_task_basis_message_id": basis,
            "task_hash_stable_count": count,
        })
    return updated, TaskBoundaryObservation(decision, basis, candidate, confirmed, 0 if confirmed else count, required_stable_count)


_CLASSIFIER_PROMPT = """你是任务边界分类器。只返回 JSON，不要解释。
判断最新用户消息相对于当前活动任务是 same、new 或 uncertain。
same：继续、追问或补充当前任务；new：不同目标、主题或交付物；不确定才用 uncertain。
格式：{"decision":"same|new|uncertain"}"""


async def classify(
    adapter,
    *,
    model: str,
    history: Sequence[Message],
    user_input: str,
    summary_state: ConversationSummaryState | None,
) -> tuple[TaskBoundaryDecision, ModelUsage]:
    recent = tuple(history[-12:])
    summary = summary_state.summary.render_markdown() if summary_state else "（暂无历史摘要）"
    request = ModelRequest(
        messages=(
            Message(role=MessageRole.SYSTEM, content=_CLASSIFIER_PROMPT),
            Message(role=MessageRole.USER, content=json.dumps({
                "active_task_hash": summary_state.active_task_hash if summary_state else None,
                "summary": summary,
                "recent_messages": [m.model_dump(mode="json") for m in recent],
                "latest_user_message": user_input,
            }, ensure_ascii=False)),
        ),
        model=model,
        max_output_tokens=64,
    )
    try:
        response = await adapter.complete(request=request)
        parsed = json.loads((response.message.content or "").strip())
        return TaskBoundaryDecision(parsed["decision"]), response.usage
    except Exception:
        return TaskBoundaryDecision.UNCERTAIN, ModelUsage()


__all__ = [
    "TaskBoundaryDecision", "TaskBoundaryObservation", "TaskBoundaryTrigger",
    "basis_message_id", "classify", "ensure_initial_task", "observe", "task_hash",
]
