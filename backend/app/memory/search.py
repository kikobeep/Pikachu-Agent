
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.model.config import Message, MessageRole
from app.memory.regular import MemoryStatus
from app.memory.search_config import (
    MemorySearchCandidate,
    MemorySearchInputs,
    MemorySearchResult,
    SearchMode,
)
from app.memory.search_index import MemorySearchIndex
from app.memory.store import MemoryStore


class MemorySearchService:
    """检索相关记忆，并根据真实记录更新候选信息。"""

    def __init__(
        self,
        store: MemoryStore,
        search_index: MemorySearchIndex,
    ) -> None:
        self._store = store
        self._search_index = search_index

    async def search(
        self,
        inputs: MemorySearchInputs,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        query = inputs.render(
            max_chars=self._search_index.configs.query_max_chars
        )

        if not query:
            return MemorySearchResult(
                query="",
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                degrade_reason="记忆检索输入为空",
            )
        try:
            result = await self._search_index.search(query, limit=limit)
            candidates: list[MemorySearchCandidate] = []

            for candidate in result.candidates:
                record = await self._store.load(candidate.memory_id)

                if record is None or record.status is not MemoryStatus.ACTIVE:
                    continue

                candidates.append(
                    replace(
                        candidate,
                        title=record.title,
                        summary=record.summary,
                        revision=record.revision,
                    )
                )

            return replace(
                result,
                candidates=tuple(candidates),
            )

        except Exception as exc:
            return MemorySearchResult(
                query=query,
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                degrade_reason=f"记忆检索失败：{type(exc).__name__}: {exc}",
            )

def recent_user_message_texts(
    history: Sequence[Message],
    *,
    limit: int = 3,
    max_chars: int = 300,
) -> tuple[str, ...]:
    """从持久历史中取最近几条用户消息（不含当前 Run 的新消息）。"""

    texts: list[str] = []
    for message in reversed(history):
        if message.role is not MessageRole.USER:
            continue
        content = (message.content or "").strip()
        if content:
            texts.append(content[:max_chars])
        if len(texts) >= limit:
            break
    return tuple(reversed(texts))
