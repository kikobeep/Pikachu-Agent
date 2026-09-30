"""把一次 Run 投影成改进证据：事件选择 + 文本渲染。"""

from __future__ import annotations

from app.agent.events import AgentEvent, AgentEventType
from app.run.config import Run

# 参与改进的工具调用/模型完成事件；流式增量与内部预算事件属于噪声，被排除。
_KEEP_EVENT_TYPES = frozenset(
    {
        AgentEventType.AGENT_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.AGENT_COMPLETED,
        AgentEventType.AGENT_FAILED,
    }
)


class DefaultEventSelector:
    """保留对改进有价值的工具调用/模型完成事件，丢弃流式增量噪声。"""

    def select(
        self,
        run: Run,
        events: tuple[AgentEvent, ...],
    ) -> tuple[AgentEvent, ...]:
        return tuple(event for event in events if event.type in _KEEP_EVENT_TYPES)


class TraceEvidenceBuilder:
    """把选中的事件渲染成紧凑、有界的一段文本证据。"""

    def __init__(self, *, max_chars: int = 20_000) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def build(self, run: Run, events: tuple[AgentEvent, ...]) -> str:
        lines: list[str] = []
        used = 0
        for event in events:
            line = self._render(event)
            if not line:
                continue
            if used + len(line) > self._max_chars:
                if lines:
                    lines.append("…")
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    @staticmethod
    def _render(event: AgentEvent) -> str:
        parts = [event.type.value]
        if event.tool_call is not None:
            args = event.tool_call.arguments
            preview = str(args) if isinstance(args, dict) else str(args)
            parts.append(f"{event.tool_call.name}({preview[:120]})")
        if event.tool_result is not None:
            parts.append("success" if event.tool_result.success else "failed")
            if not event.tool_result.success and event.tool_result.error:
                parts.append(str(event.tool_result.error)[:200])
        if event.error is not None:
            parts.append(str(event.error)[:200])
        return " ".join(parts)


__all__ = ["DefaultEventSelector", "TraceEvidenceBuilder"]
