"""模型请求的分层缓存布局。

这个模块只描述请求的缓存边界，不改变上下文压缩结果。具体模型适配器可以在
后续根据这些边界接入 provider-specific 的 prompt cache breakpoint。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Sequence

from app.context.blocks import MessageBlock, partition_messages
from app.model.config import Message, MessageRole
from app.tools.config import ToolDefinition


_LAYOUT_VERSION = 1
_RECENT_BLOCK_LIMIT = 3


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _message_payload(message: Message) -> dict[str, object]:
    return message.model_dump(mode="json", exclude_none=True)


def _prefix_hash(messages: Sequence[Message], end: int) -> str:
    return _digest([_message_payload(message) for message in messages[:end]])


def _block_ranges(
    messages: Sequence[Message],
) -> tuple[tuple[MessageBlock, int, int], ...]:
    ranges: list[tuple[MessageBlock, int, int]] = []
    position = 0
    for block in partition_messages(messages):
        end = position + len(block)
        ranges.append((block, position, end))
        position = end
    return tuple(ranges)


def _recent_breakpoints(
    messages: Sequence[Message],
    fixed_end: int,
) -> tuple[int, tuple[int, ...]]:
    """选择最近的完整消息块作为近期层，并返回其前缀断点。"""

    ranges = _block_ranges(messages)
    eligible = [
        item
        for item in ranges
        if item[2] > fixed_end and item[0].block_type.value != "system"
    ]
    selected = eligible[-_RECENT_BLOCK_LIMIT:]
    if not selected:
        return fixed_end, ()
    return selected[0][1], tuple(item[2] for item in selected)


def _first_changed_layer(
    previous: CacheLayout | None,
    *,
    tool_hash: str,
    fixed_prefix_hash: str,
    session_anchor_hash: str,
    session_prefix_unchanged: bool,
    recent_prefix_hashes: tuple[str, ...],
    volatile_tail_hash: str,
) -> str:
    if previous is None:
        return "cold_start"
    if (
        previous.tool_hash != tool_hash
        or previous.fixed_prefix_hash != fixed_prefix_hash
    ):
        return "fixed"
    if (
        previous.session_anchor_hash != session_anchor_hash
        or not session_prefix_unchanged
    ):
        return "session"
    if previous.recent_prefix_hashes != recent_prefix_hashes:
        return "recent"
    if previous.volatile_tail_hash != volatile_tail_hash:
        return "volatile"
    return "unchanged"


@dataclass(frozen=True)
class CacheLayout:
    """一次模型请求的四层缓存布局及其变化位置。"""

    version: int
    fixed_message_count: int
    session_message_count: int
    recent_start_message: int
    recent_breakpoint_positions: tuple[int, ...]
    volatile_start_message: int
    tool_hash: str
    fixed_prefix_hash: str
    session_prefix_hash: str
    session_anchor_hash: str
    recent_prefix_hashes: tuple[str, ...]
    volatile_tail_hash: str
    first_changed_layer: str

    def event_fields(self) -> dict[str, object]:
        """返回可直接写入 AgentEvent 的无正文诊断字段。"""

        return {
            "cache_layout_version": self.version,
            "cache_first_changed_layer": self.first_changed_layer,
            "cache_fixed_message_count": self.fixed_message_count,
            "cache_session_message_count": self.session_message_count,
            "cache_recent_start_message": self.recent_start_message,
            "cache_recent_breakpoint_positions": self.recent_breakpoint_positions,
            "cache_volatile_start_message": self.volatile_start_message,
            "cache_tools_hash": self.tool_hash,
            "cache_fixed_prefix_hash": self.fixed_prefix_hash,
            "cache_session_prefix_hash": self.session_prefix_hash,
            "cache_recent_prefix_hashes": self.recent_prefix_hashes,
            "cache_volatile_tail_hash": self.volatile_tail_hash,
        }


def build_cache_layout(
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition],
    *,
    previous: CacheLayout | None = None,
) -> CacheLayout:
    """为最终发送的请求计算固定、会话、近期、易变尾部四层布局。

    固定层是请求开头连续的未命名 system 消息；会话层延伸到近期层之前；近期层
    由最近三个完整的非 system 消息块组成；最后一个断点之后的内容视为
    易变尾部。这里使用消息数量作为断点单位，避免把用户正文写入事件。
    """

    message_count = len(messages)
    fixed_end = int(
        bool(
            message_count > 0
            and messages[0].role is MessageRole.SYSTEM
            and messages[0].name is None
        )
    )

    recent_start, recent_breakpoints = _recent_breakpoints(messages, fixed_end)
    recent_start = max(fixed_end, min(recent_start, message_count))
    volatile_start = (
        recent_breakpoints[-1]
        if recent_breakpoints
        else recent_start
    )
    volatile_start = max(recent_start, min(volatile_start, message_count))

    tool_hash = _digest(
        [tool.model_dump(mode="json", exclude_none=True) for tool in tools]
    )
    fixed_prefix_hash = _prefix_hash(messages, fixed_end)
    session_prefix_hash = _prefix_hash(messages, recent_start)
    session_anchor_hash = _digest(
        _message_payload(messages[fixed_end])
        if fixed_end < message_count
        else None
    )
    recent_prefix_hashes = tuple(
        _prefix_hash(messages, position) for position in recent_breakpoints
    )
    volatile_tail_hash = _digest(
        [_message_payload(message) for message in messages[volatile_start:]]
    )
    session_prefix_unchanged = (
        previous is not None
        and previous.recent_start_message <= message_count
        and _prefix_hash(messages, previous.recent_start_message)
        == previous.session_prefix_hash
    )

    return CacheLayout(
        version=_LAYOUT_VERSION,
        fixed_message_count=fixed_end,
        session_message_count=max(0, recent_start - fixed_end),
        recent_start_message=recent_start,
        recent_breakpoint_positions=recent_breakpoints,
        volatile_start_message=volatile_start,
        tool_hash=tool_hash,
        fixed_prefix_hash=fixed_prefix_hash,
        session_prefix_hash=session_prefix_hash,
        session_anchor_hash=session_anchor_hash,
        recent_prefix_hashes=recent_prefix_hashes,
        volatile_tail_hash=volatile_tail_hash,
        first_changed_layer=_first_changed_layer(
            previous,
            tool_hash=tool_hash,
            fixed_prefix_hash=fixed_prefix_hash,
            session_anchor_hash=session_anchor_hash,
            session_prefix_unchanged=session_prefix_unchanged,
            recent_prefix_hashes=recent_prefix_hashes,
            volatile_tail_hash=volatile_tail_hash,
        ),
    )


__all__ = ["CacheLayout", "build_cache_layout"]
