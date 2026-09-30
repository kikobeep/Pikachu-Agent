"""供模型按需检索原始会话历史。"""

from __future__ import annotations

from typing import Any

from app.tools.config import BaseTool, ToolDefinition, ToolPermission
from app.tools.config import ToolExecutionContext
from app.tools.register import ToolRegistry

from .models import ConversationMessageRecord
from .store import ConversationStore


class HistorySearchTool(BaseTool):
    """搜索当前会话数据库中的完整原始消息。"""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_search",
            record_output=False,
            description=(
                "搜索当前会话数据库中的完整原始消息。滚动摘要遗漏了用户约束、"
                "决定或旧讨论细节时使用；结果中的 sequence 可交给 history_read。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词。"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("history_search requires conversation context")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        limit = _integer(arguments.get("limit", 10), "limit")
        records = await self._store.search_messages(
            conversation_id,
            query,
            limit=limit,
        )
        return {
            "query": query,
            "count": len(records),
            "results": [_public_record(record, max_chars=500) for record in records],
        }


class HistoryReadTool(BaseTool):
    """读取指定消息前后的原始会话窗口。"""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_read",
            record_output=False,
            description=(
                "按 sequence 读取当前会话的一段原始消息窗口。只在 history_search"
                "已定位到相关消息，或明确知道消息序号时使用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sequence": {"type": "integer", "minimum": 0},
                    "before": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 2,
                    },
                    "after": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 2,
                    },
                },
                "required": ["sequence"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("history_read requires conversation context")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        sequence = _integer(arguments.get("sequence"), "sequence")
        before = _integer(arguments.get("before", 2), "before")
        after = _integer(arguments.get("after", 2), "after")
        records = await self._store.load_message_window(
            conversation_id,
            sequence,
            before=before,
            after=after,
        )
        return {
            "sequence": sequence,
            "count": len(records),
            "messages": [_public_record(record, max_chars=4000) for record in records],
        }


def register_history_tools(
    registry: ToolRegistry,
    store: ConversationStore,
) -> None:
    registry.register(HistorySearchTool(store), on_demand=True)
    registry.register(HistoryReadTool(store), on_demand=True)


def _public_record(
    record: ConversationMessageRecord,
    *,
    max_chars: int,
) -> dict[str, Any]:
    content = record.message.content
    if content is not None and len(content) > max_chars:
        content = f"{content[:max_chars]}…"
    return {
        "sequence": record.sequence,
        "role": record.message.role.value,
        "name": record.message.name,
        "tool_call_id": record.message.tool_call_id,
        "content": content,
        "created_at": record.created_at.isoformat(),
    }


def _require_conversation(context: ToolExecutionContext) -> str:
    if not context.conversation_id:
        raise ValueError("history tool requires conversation context")
    return context.conversation_id


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer")
    return value


__all__ = ["HistoryReadTool", "HistorySearchTool", "register_history_tools"]
