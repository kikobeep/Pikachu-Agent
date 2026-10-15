"""Light 模式的上下文渲染和模型输出解析。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .models import (
    LightModelEnvelope,
    LightStateDelta,
    LightStateNode,
    LightStateOperation,
    LightStateType,
    LightToolCall,
)


def render_historical_index(nodes: tuple[LightStateNode, ...]) -> str:
    if not nodes:
        return "<historical_state_index />"
    lines = ["<historical_state_index>"]
    for node in nodes:
        evidence = ", ".join(node.evidence_ids) or "none"
        relations = ", ".join(
            f"{relation.relation}->{relation.target_state_id}"
            for relation in node.relations
        ) or "none"
        lines.append(
            f'- {node.state_id} [{node.state_type.value}] {node.summary} '
            f'(revision={node.revision}, evidence={evidence}, relations={relations})'
        )
    lines.append("</historical_state_index>")
    return "\n".join(lines)


def render_active_state(nodes: tuple[LightStateNode, ...]) -> str:
    if not nodes:
        return "<active_state />"
    lines = ["<active_state>"]
    for node in nodes:
        value = json.dumps(node.value, ensure_ascii=False, sort_keys=True)
        evidence = ", ".join(node.evidence_ids) or "none"
        relations = ", ".join(
            f"{relation.relation}->{relation.target_state_id}"
            for relation in node.relations
        ) or "none"
        lines.append(
            f'- {node.state_id} [{node.state_type.value}] {node.summary}; '
            f'value={value}; evidence={evidence}; relations={relations}; '
            f'revision={node.revision}'
        )
    lines.append("</active_state>")
    return "\n".join(lines)


def render_observation(observation: str | None) -> str:
    if not observation:
        return "<latest_observation />"
    return f"<latest_observation>\n{observation}\n</latest_observation>"


def parse_model_output(
    content: str | None,
    *,
    native_tool_calls: tuple[Any, ...] = (),
) -> LightModelEnvelope:
    """解析 JSON envelope；模型偶尔返回普通文本时保留为最终回答。"""

    text = (content or "").strip()
    native_calls = tuple(
        LightToolCall.model_validate(
            call.model_dump() if hasattr(call, "model_dump") else call
        )
        for call in native_tool_calls
    )
    if not text:
        return LightModelEnvelope(tool_calls=native_calls)
    try:
        payload = _decode_json_object(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return LightModelEnvelope(
            assistant_message=text,
            tool_calls=native_calls,
        )
    try:
        envelope = LightModelEnvelope.model_validate(payload)
    except ValidationError:
        # 允许模型只返回 assistant_message，同时不让协议瑕疵破坏整个 Run。
        return LightModelEnvelope(
            assistant_message=text,
            tool_calls=native_calls,
        )
    if native_tool_calls:
        envelope = envelope.model_copy(update={"tool_calls": native_calls})
    return envelope


def normalize_content_tool_calls(envelope: LightModelEnvelope) -> LightModelEnvelope:
    """把 JSON 中的 fallback tool_calls 规范为统一 ToolCall 之前的形态。"""

    normalized = []
    for index, call in enumerate(envelope.tool_calls):
        if not call.name.strip():
            raise ValueError("light tool call name cannot be empty")
        normalized.append(
            call.model_copy(
                update={"id": call.id or f"light-call-{index + 1}"}
            )
        )
    return envelope.model_copy(update={"tool_calls": tuple(normalized)})


def validate_state_delta(delta: LightStateDelta) -> None:
    if delta.operation is LightStateOperation.ADD:
        if delta.state_type is None or not (delta.summary or "").strip():
            raise ValueError("add state_delta requires state_type and summary")
    elif delta.operation in {
        LightStateOperation.UPDATE,
        LightStateOperation.REMOVE,
    } and not delta.state_id:
        raise ValueError(f"{delta.operation.value} state_delta requires state_id")
    if delta.state_type is not None and delta.state_type not in set(LightStateType):
        raise ValueError("unsupported light state type")


def _decode_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    if start < 0:
        raise ValueError("light response does not contain a JSON object")
    payload, _ = decoder.raw_decode(candidate[start:])
    if not isinstance(payload, dict):
        raise ValueError("light response must be a JSON object")
    return payload


__all__ = [
    "normalize_content_tool_calls",
    "parse_model_output",
    "render_active_state",
    "render_historical_index",
    "render_observation",
    "validate_state_delta",
]
