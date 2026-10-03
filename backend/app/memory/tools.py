
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from app.memory.prompt import MEMORY_WRITE_POLICY
from app.tools.config import BaseTool, ToolDefinition
from app.tools.register import ToolRegistry

from .manager import MemoryManager
from .search_config import MemorySearchInputs

from ..tools.config import ToolExecutionContext

DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES = frozenset(
    {"memory_list", "core_memory_update", "core_memory_remove"}
)

MEMORY_SEARCH_TOOL_NAME = "memory_search"


class MemoryReadTool(BaseTool):
    """读取一条完整长期记忆。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_read",
            record_output=False,
            description=(
                "读取一条完整长期记忆。memory_search 只返回摘要片段；如果搜索"
                "返回候选，必须使用候选的 memory_id 调用本工具读取完整内容，再回答。"
                "每次读取会记录访问次数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "要读取的 Memory ID，例如 M001。",
                    },
                },
                "required": ["memory_id"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        record = await self._manager.read(memory_id)
        if record is None:
            return {"found": False, "memory_id": memory_id}
        return {
            "found": True,
            "id": record.id,
            "title": record.title,
            "revision": record.revision,
            "content": record.render_full(),
        }


class MemorySearchTool(BaseTool):

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_search",
            record_output=False,
            description=(
                "在当前 active 长期记忆中做混合检索（关键词 + 语义），用于初始"
                "自动召回遗漏、或 Run 中途切换话题时补充候选。只返回 id、标题、"
                "摘要、revision 与最相关片段，不返回完整正文；需要依赖完整记忆时"
                "必须继续调用 memory_read。本搜索不计入记忆读取次数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询：任务描述、关键词或主题短语。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "返回候选数量上限，默认 5。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        limit = arguments.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("'limit' must be a positive integer")
        # Native agent-loop search must obey the benchmark/runtime top-k even
        # if the model requests a larger limit in its tool arguments.
        configured_limit = self._manager.search_configs.top_k
        # The runtime owns the candidate budget. In hybrid mode this is the
        # union of one vector candidate and one BM25 candidate, not a model-
        # supplied arbitrary limit.
        limit = configured_limit
        result = await self._manager.search(
            MemorySearchInputs(user_message=query),
            limit=limit,
        )
        if result.mode.value == "unavailable":
            return {
                "available": False,
                "query": query,
                "reason": result.degrade_reason or "search index unavailable",
                "hint": "fall back to Memory Index cues and memory_read",
            }
        # Apply the configured vector gate again at the tool boundary.  The
        # search index already applies it while building candidates, but this
        # guard prevents weak vector hits from leaking into the model when a
        # different search path or a future index implementation is used.
        min_similarity = self._manager.search_configs.min_vector_similarity
        candidates = tuple(
            candidate
            for candidate in result.candidates
            if not self._manager.search_configs.vector_only
            or (
                candidate.vector_similarity is not None
                and candidate.vector_similarity >= min_similarity
            )
        )
        return {
            "available": True,
            "query": query,
            "mode": result.mode.value,
            "results": [
                {
                    "memory_id": candidate.memory_id,
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "revision": candidate.revision,
                    "snippet": candidate.snippet,
                    "vector_similarity": candidate.vector_similarity,
                    "vector_rank": candidate.vector_rank,
                    "bm25_rank": candidate.text_rank,
                    "retrieval_source": (
                        "hybrid"
                        if candidate.matched_by_vector and candidate.matched_by_text
                        else "vector"
                        if candidate.matched_by_vector
                        else "bm25"
                    ),
                    "matched_by": [
                        source
                        for source, hit in (
                            ("vector", candidate.matched_by_vector),
                            ("fts", candidate.matched_by_text),
                        )
                        if hit
                    ],
                }
                for candidate in candidates
            ],
        }


class MemoryListTool(BaseTool):
    """列出当前 active 长期记忆的简要信息。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_list",
            record_output=False,
            description=(
                "列出当前 active 长期记忆的 id、标题与摘要（Recall Cue）。"
                "不返回完整正文；需要详情时用 memory_read。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        records = await self._manager.list()
        return {
            "memories": [
                {
                    "id": record.id,
                    "title": record.title,
                    "summary": record.summary,
                }
                for record in records
            ]
        }


class MemoryCreateTool(BaseTool):
    """创建一条普通长期记忆。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_create",
            record_output=False,
            description=(
                "创建一条长期记忆。对未来跨会话仍有价值的信息才创建；"
                f"写入条件：{MEMORY_WRITE_POLICY}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "记忆标题。",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Recall Cue，用于 Index 中提示模型何时读取。",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "记忆完整正文。只输出正文内容，不要重复输出标题、"
                            "## Summary、## Memory 或其他 Markdown 外壳；标题和摘要"
                            "由系统单独渲染。"
                        ),
                    },
                },
                "required": ["title", "summary", "content"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = arguments.get("title")
        summary = arguments.get("summary")
        content = arguments.get("content")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("'summary' must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("'content' must be a non-empty string")
        record = await self._manager.create_if_capacity(
            title=title,
            summary=summary,
            content=content,
        )
        if record is None:
            raise ValueError("active memory capacity is full")
        result: dict[str, Any] = {
            "id": record.id,
            "title": record.title,
            "summary": record.summary,
            "revision": record.revision,
        }
        return result


class MemoryUpdateTool(BaseTool):
    """修正已有长期记忆。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_update",
            record_output=False,
            description=(
                "基于最近一次读取的 revision，替换已有长期记忆的标题、Recall "
                "Cue 和完整正文。新信息属于旧主题时优先 update，避免重复记忆。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "要更新的 Memory ID。",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "修正后的完整正文。只输出正文内容，不要重复输出标题、"
                            "## Summary、## Memory 或其他 Markdown 外壳；标题和摘要"
                            "由系统单独渲染。"
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "修正后的完整标题。",
                    },
                    "summary": {
                        "type": "string",
                        "description": "修正后的完整 Recall Cue。",
                    },
                    "expected_revision": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "最近一次 memory_read 返回的 revision。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "更新原因（留痕）。",
                    },
                },
                "required": [
                    "memory_id",
                    "title",
                    "summary",
                    "content",
                    "reason",
                    "expected_revision",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        title = arguments.get("title")
        summary = arguments.get("summary")
        content = arguments.get("content")
        reason = arguments.get("reason")
        expected_revision = arguments.get("expected_revision")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("'title' must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("'summary' must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("'content' must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("'reason' must be a non-empty string")
        if not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("'expected_revision' must be a positive integer")
        record = await self._manager.update_if_revision(
            memory_id,
            expected_revision=expected_revision,
            title=title,
            summary=summary,
            content=content,
            reason=reason,
        )
        return {
            "id": record.id,
            "title": record.title,
            "summary": record.summary,
            "revision": record.revision,
            "updated": True,
        }


class MemoryArchiveTool(BaseTool):
    """把过时长期记忆归档。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_archive",
            record_output=False,
            description=(
                "把一条过时或不再需要的长期记忆归档。归档后不再出现在 Memory "
                "Index 中，也不会进入模型上下文。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "要归档的 Memory ID。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "归档原因（留痕）。",
                    },
                },
                "required": ["memory_id", "reason"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        reason = arguments.get("reason")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("'memory_id' must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("'reason' must be a non-empty string")
        record = await self._manager.archive(memory_id, reason=reason)
        return {"id": record.id, "status": record.status.value}


class CoreMemoryUpdateTool(BaseTool):
    """根据当前用户的明确长期陈述，按 key 创建或更新 Core Memory。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="core_memory_update",
            record_output=False,
            description=(
                "按稳定 key 创建或更新一条 Core Memory。调用前先判断：即使当前 "
                "Run 与当前项目或仓库完全无关，这条信息是否仍应在每次 Run 常驻？"
                "只有当前用户明确表达的稳定身份、全局长期偏好或全局安全/隐私约束"
                "才属于 Core。项目或仓库的架构、技术选型、路径、实现约束和历史决定"
                "属于 Ordinary Memory，由 Run 后的 Memory Reflection 处理；当前任务"
                "状态属于 Task，可复用流程属于 Skills。必须逐字复制当前用户消息中的"
                "明确原话作为 explicit_user_statement；不要根据推断调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "稳定的小写点分 key，例如 communication.language。"
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "应在每次 Run 常驻的精简长期事实或约束。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么该信息属于 Core，以及本次更新原因。",
                    },
                    "explicit_user_statement": {
                        "type": "string",
                        "description": "从当前用户消息逐字复制的明确长期陈述。",
                    },
                },
                "required": [
                    "key",
                    "value",
                    "reason",
                    "explicit_user_statement",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("core_memory_update requires the current user message")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        key = _required_string(arguments, "key")
        value = _required_string(arguments, "value")
        reason = _required_string(arguments, "reason")
        statement = _required_string(arguments, "explicit_user_statement")
        user_input = context.user_input
        if not user_input or statement not in user_input:
            raise ValueError(
                "'explicit_user_statement' must be copied exactly from the "
                "current user message"
            )
        entry, created = await self._manager.upsert_core(
            key=key,
            value=value,
            reason=reason,
            source_statement=statement,
        )
        return {
            "key": entry.key,
            "value": entry.value,
            "created": created,
            "updated": not created,
        }


class CoreMemoryRemoveTool(BaseTool):
    """根据当前用户明确要求移除一个按 key 管理的 Core 条目。"""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="core_memory_remove",
            record_output=False,
            description=(
                "移除一个不再成立的 Core Memory 条目。只能在当前用户明确撤销"
                "稳定身份、全局长期偏好或全局安全/隐私约束时调用；必须逐字复制"
                "当前用户"
                "消息中的撤销原话，不要根据推断或旧消息移除。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "要移除的稳定小写点分 key。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "移除原因（会保留在 Tool/Trace 记录中）。",
                    },
                    "explicit_user_statement": {
                        "type": "string",
                        "description": "从当前用户消息逐字复制的明确撤销原话。",
                    },
                },
                "required": ["key", "reason", "explicit_user_statement"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("core_memory_remove requires the current user message")

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        key = _required_string(arguments, "key")
        _required_string(arguments, "reason")
        statement = _required_string(arguments, "explicit_user_statement")
        user_input = context.user_input
        if not user_input or statement not in user_input:
            raise ValueError(
                "'explicit_user_statement' must be copied exactly from the "
                "current user message"
            )
        removed = await self._manager.remove_core(key)
        return {"key": removed.key, "removed": True}


def register_memory_tools(
    registry: ToolRegistry,
    manager: MemoryManager,
) -> None:
    registry.register(MemoryReadTool(manager))
    registry.register(MemorySearchTool(manager))
    registry.register(MemoryListTool(manager))
    registry.register(CoreMemoryUpdateTool(manager))
    registry.register(CoreMemoryRemoveTool(manager))


def register_memory_write_tools(
    registry: ToolRegistry,
    manager: MemoryManager,
) -> None:
    """为离线测试或未来内部 Agent 注册普通 Memory mutation 工具。"""

    registry.register(MemoryCreateTool(manager))
    registry.register(MemoryUpdateTool(manager))
    registry.register(MemoryArchiveTool(manager))


def _required_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string")
    return value.strip()


__all__ = [
    "DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES",
    "MEMORY_SEARCH_TOOL_NAME",
    "CoreMemoryRemoveTool",
    "CoreMemoryUpdateTool",
    "MemoryArchiveTool",
    "MemoryCreateTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemorySearchTool",
    "MemoryUpdateTool",
    "register_memory_tools",
    "register_memory_write_tools",
]
