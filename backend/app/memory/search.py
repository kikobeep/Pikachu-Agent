
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

from app.model.config import Message, MessageRole
from app.memory.regular import MemoryStatus, RegularMemoryRecord
from app.memory.search_config import (
    MemorySearchConfig,
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
            records = {}

            for candidate in result.candidates:
                record = await self._store.load(candidate.memory_id)

                if record is None or record.status is not MemoryStatus.ACTIVE:
                    continue
                records[record.id] = record

                candidates.append(
                    replace(
                        candidate,
                        title=record.title,
                        summary=record.summary,
                        revision=record.revision,
                    )
                )

            candidates = _rerank_regular_memories(
                candidates,
                records=records,
                config=self._search_index.configs,
                limit=limit or self._search_index.configs.top_k,
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


def _rerank_regular_memories(
    candidates: Sequence[MemorySearchCandidate],
    *,
    records: dict[str, RegularMemoryRecord],
    config: MemorySearchConfig,
    limit: int,
) -> tuple[MemorySearchCandidate, ...]:
    """按相关性优先，再结合遗忘曲线和重要性重排普通记忆。

    Core Memory 不经过这里；它由 MemoryManager.memory_context() 单独常驻注入。
    ``retention`` 是 MemoryBank 风格的工程化实现：年龄按天计算，记忆被
    memory_read 访问的次数增加后，等效记忆强度随之增加。
    """

    if not candidates:
        return ()
    valid_records = records
    if not valid_records:
        return tuple(candidates[:limit])

    max_rrf = max((candidate.rrf_score for candidate in candidates), default=0.0)
    scored: list[MemorySearchCandidate] = []
    now = datetime.now(UTC)
    for candidate in candidates:
        record = valid_records.get(candidate.memory_id)
        if record is None:
            continue

        relevance = _relevance_score(candidate, max_rrf)
        retention = _retention_score(record, config, now)
        importance = record.importance
        weight_total = (
            config.relevance_weight
            + config.retention_weight
            + config.importance_weight
        )
        rerank_score = (
            config.relevance_weight * relevance
            + config.retention_weight * retention
            + config.importance_weight * importance
        ) / weight_total
        scored.append(
            replace(
                candidate,
                relevance_score=relevance,
                retention_score=retention,
                importance_score=importance,
                rerank_score=rerank_score,
            )
        )
    scored.sort(key=lambda candidate: (-candidate.rerank_score, candidate.memory_id))
    return tuple(scored[:limit])


def _relevance_score(candidate: MemorySearchCandidate, max_rrf: float) -> float:
    """把 BM25+向量的 RRF 混合分数归一化为 [0, 1] 的相关性分数。"""

    relevance_score = (
        candidate.rrf_score / max_rrf if max_rrf > 0.0 else 0.0
    )
    # RRF 已经融合了 BM25 和向量召回；这里不再重复叠加
    # vector_similarity，避免向量信号被计算两次。
    return max(0.0, min(1.0, relevance_score))


def _retention_score(record, config, now: datetime) -> float:
    """计算 MemoryBank 风格的记忆保留分数。"""

    age_days = max(
        0.0,
        (now - record.last_accessed_at).total_seconds() / 86_400.0,
    )
    # MemoryBank 中每次 recall 都令 S += 1；这里用 memory_read 的访问次数
    # 作为 recall 次数，并乘以可配置的基础时间尺度。
    strength_days = config.retention_strength_days * (1.0 + record.access_count)
    return math.exp(-age_days / strength_days)

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
