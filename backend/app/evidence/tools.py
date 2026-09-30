"""供模型按需检索、分页回读不可变工具证据。"""

from __future__ import annotations

from typing import Any

from app.tools.config import BaseTool, ToolDefinition, ToolPermission
from app.tools.config import ToolExecutionContext
from app.tools.register import ToolRegistry

from .store import EvidenceStore


class EvidenceSearchTool(BaseTool):
    """在当前会话的原始工具输出中搜索关键词。"""

    def __init__(self, store: EvidenceStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evidence_search",
            record_output=False,
            description=(
                "搜索当前会话中已归档的完整工具原始输出。上下文里的工具结果"
                "被截断、清理或摘要后，用它重新定位原始证据；只返回片段，"
                "需要正文再调用 evidence_read。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词。"},
                    "tool_name": {
                        "type": "string",
                        "description": "可选，限定原工具名。",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "可选，限定关联任务 ID。",
                    },
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
        raise ValueError("evidence_search requires conversation context")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        hits = await self._store.search(
            conversation_id=conversation_id,
            query=query,
            tool_name=_optional_text(arguments.get("tool_name")),
            task_id=_optional_text(arguments.get("task_id")),
            limit=_integer(arguments.get("limit", 10), "limit"),
        )
        return {
            "query": query,
            "count": len(hits),
            "results": [
                {
                    "evidence_id": hit.record.id,
                    "tool_name": hit.record.tool_name,
                    "run_id": hit.record.run_id,
                    "content_chars": hit.record.content_chars,
                    "task_id": hit.record.task_id,
                    "task_step_id": hit.record.task_step_id,
                    "created_at": hit.record.created_at.isoformat(),
                    "snippet": hit.snippet,
                }
                for hit in hits
            ],
        }


class EvidenceReadTool(BaseTool):
    """按稳定 Evidence ID 分页读取完整原始输出。"""

    def __init__(self, store: EvidenceStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="evidence_read",
            record_output=False,
            description=(
                "按 Evidence ID 读取当前会话的一段原始工具输出。内容很大时使用"
                " offset/limit 分页；跨会话证据统一表现为不存在。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "完整 ID 或至少 4 位的唯一前缀。",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12000,
                        "default": 6000,
                    },
                },
                "required": ["evidence_id"],
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("evidence_read requires conversation context")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        conversation_id = _require_conversation(context)
        identifier = arguments.get("evidence_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("'evidence_id' must be a non-empty string")
        offset = _integer(arguments.get("offset", 0), "offset")
        limit = _integer(arguments.get("limit", 6000), "limit")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if limit < 1 or limit > 12000:
            raise ValueError("limit must be between 1 and 12000")
        document = await self._store.resolve(
            identifier,
            conversation_id=conversation_id,
        )
        if document is None:
            return {"found": False, "evidence_id": identifier}
        content = document.content[offset : offset + limit]
        next_offset = offset + len(content)
        return {
            "found": True,
            "evidence_id": document.record.id,
            "tool_name": document.record.tool_name,
            "run_id": document.record.run_id,
            "task_id": document.record.task_id,
            "task_step_id": document.record.task_step_id,
            "sha256": document.record.sha256,
            "content_chars": document.record.content_chars,
            "offset": offset,
            "content": content,
            "next_offset": (
                next_offset
                if next_offset < document.record.content_chars
                else None
            ),
        }


def register_evidence_tools(
    registry: ToolRegistry,
    store: EvidenceStore,
) -> None:
    registry.register(EvidenceSearchTool(store), on_demand=True)
    registry.register(EvidenceReadTool(store), on_demand=True)


def _require_conversation(context: ToolExecutionContext) -> str:
    if not context.conversation_id:
        raise ValueError("evidence tool requires conversation context")
    return context.conversation_id


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional filter must be a string")
    return value.strip() or None


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer")
    return value


__all__ = ["EvidenceReadTool", "EvidenceSearchTool", "register_evidence_tools"]
