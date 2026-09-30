
from __future__ import annotations

import asyncio
import logging
from array import array
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import numpy as np
import hashlib
from app.memory.embedding import EmbeddingAdapter
from app.memory.regular import MemoryStatus, RegularMemoryRecord
from app.memory.search_config import (
    MemorySearchCandidate,
    MemorySearchConfig,
    MemorySearchResult,
    SearchMode,
)

logger = logging.getLogger(__name__)

_SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    memory_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dim INTEGER,
    embedding BLOB,
    PRIMARY KEY (memory_id, chunk_index)
);
"""
_RRF = 6


@dataclass(frozen=True, slots=True)
class _ChunkHit:
    """单路检索返回的一个 Chunk 命中。"""

    memory_id: str
    chunk_index: int
    title: str
    text: str
    score: float

class MemorySearchIndex:
    def __init__(
        self,
        database_path: Path | str,
        *,
        embedding: EmbeddingAdapter | None,
        configs: MemorySearchConfig | None = None,
    ):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._bm25_available = False
        self.embedding = embedding
        self.configs = configs if configs is not None else MemorySearchConfig()
        self._write_lock = asyncio.Lock()
    

    async def initialize(self) -> None:
        try:
            # 确保数据库里的表是否已准备好
            await self._check_schema()
        except aiosqlite.Error:
            # 损坏或结构过期的投影文件：直接删除后重建一次。
            logger.warning(
                "memory search index unreadable, rebuilding: %s",
                self.database_path,
            )
            self.database_path.unlink(missing_ok=True)
            await self._check_schema()
        self._initialized = True
    

    async def _check_schema(self) -> None:
        async with self._connect() as database:
            await database.execute("PRAGMA journal_mode=WAL")
            await database.executescript(_SCHEMA_CHUNKS)
            await self._ensure_search(database)
            await database.commit()
    


    async def _ensure_search(
        self,
        database: aiosqlite.Connection,
    ) -> None:
        self._bm25_available = False
        for tokenizer in ("trigram", "unicode61"):
            try:
                await database.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_text USING fts5("
                    "text, memory_id UNINDEXED, chunk_index UNINDEXED, "
                    f"tokenize='{tokenizer}')"
                )
                await database.commit()
                self._bm25_available = True
                break
            except aiosqlite.Error as exc:
                logger.info("fts5 tokenizer %s unavailable: %s", tokenizer, exc)

    
    async def reconcile(
        self,
        records: tuple[RegularMemoryRecord,...],
        *,
        generate_embeddings: bool = True
    ):
        if not self._initialized:
            return
        active_ids = {
            record.id for record in records if record.status is MemoryStatus.ACTIVE
        }
        async with self._write_lock:
            async with self._connect() as database:
                indexed_ids = {
                    row[0]
                    for row in await database.execute_fetchall(
                        "SELECT DISTINCT memory_id FROM memory_chunks"
                    )
                }
            # 删除旧记忆
            for stale_id in sorted(indexed_ids - active_ids):
                await self._remove_rows(stale_id)
            
            # 更新新记忆
            for record in records:
                if record.status is not MemoryStatus.ACTIVE:
                    continue
                await self._upsert_rows(
                    record,
                    generate_embeddings=generate_embeddings,
                )


    async def generate_embeddings(self) -> None:
        if not self._initialized or self.embedding is None:
            return
        model_name = self._embedding_model_name
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, text, text_sha256 "
                    "FROM memory_chunks WHERE embedding IS NULL "
                    "OR embedding_model IS NULL OR embedding_model != ?",
                    (model_name,),
                )
            if not rows:
                self._embeddings_ready = True
                return
            texts = tuple(str(row[2]) for row in rows)
            vectors = await self._embed_texts(texts)
            if vectors is None:
                return
            if len(vectors) != len(rows):
                logger.warning(
                    "memory embedding response count mismatch: expected=%s actual=%s",
                    len(rows),
                    len(vectors),
                )
                return
            prepared: list[tuple[bytes, int, str, int, str]] = []
            for row, vector in zip(rows, vectors, strict=True):
                normalized, blob = _normalize_vector(vector)
                prepared.append(
                    (blob, len(normalized), str(row[0]), int(row[1]), str(row[3]))
                )
            async with self._write_lock:
                async with self._connect() as database:
                    for blob, dimensions, memory_id, chunk_index, digest in prepared:
                        await database.execute(
                            "UPDATE memory_chunks SET embedding_model = ?, "
                            "embedding_dim = ?, embedding = ? WHERE memory_id = ? "
                            "AND chunk_index = ? AND text_sha256 = ?",
                            (
                                model_name,
                                dimensions,
                                blob,
                                memory_id,
                                chunk_index,
                                digest,
                            ),
                        )
                    await database.commit()
            self._embeddings_ready = True
        except Exception as exc:
            logger.warning("memory embedding backfill failed: %s", exc)

    async def _remove_rows(self, memory_id: str) -> None:
        try:
            async with self._connect() as database:
                await database.execute(
                    "DELETE FROM memory_chunks WHERE memory_id = ?",
                    (memory_id,),
                )
                if self._bm25_available:
                    await database.execute(
                        "DELETE FROM memory_text WHERE memory_id = ?",
                        (memory_id,),
                    )
                await database.commit()
        except aiosqlite.Error as exc:
            logger.warning(
                "memory search index remove failed for %s: %s",
                memory_id,
                exc,
            )
    

    async def _upsert_rows(
        self,
        record: RegularMemoryRecord,
        *,
        generate_embeddings: bool = True,
    ) -> None:
        chunks = chunk_memory_text(record, settings=self.configs)
        model_name = self._embedding_model_name
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT chunk_index, text_sha256, embedding_model, embedding "
                    "FROM memory_chunks WHERE memory_id = ?",
                    (record.id,),
                )
                existing = {row[0]: (row[1], row[2], row[3]) for row in rows}
                reusable: list[int] = []
                pending: list[tuple[int, str, str]] = []
                for index, (text, digest) in enumerate(chunks):
                    current = existing.get(index)
                    if (
                        current is not None
                        and current[0] == digest
                        and current[1] == model_name
                        and (current[2] is not None or self.embedding is None)
                    ):
                        # 内容、模型与向量都未变化：只刷新 revision。
                        reusable.append(index)
                    else:
                        pending.append((index, text, digest))
                for index in existing:
                    if index >= len(chunks):
                        await self._delete_chunk(database, record.id, index)
                if reusable:
                    placeholders = ",".join("?" * len(reusable))
                    await database.execute(
                        "UPDATE memory_chunks SET revision = ? "
                        f"WHERE memory_id = ? AND chunk_index IN ({placeholders})",
                        (record.revision, record.id, *reusable),
                    )
                if pending:
                    vectors = (
                        await self._embed_texts(tuple(item[1] for item in pending))
                        if generate_embeddings
                        else None
                    )
                    for position, (index, text, digest) in enumerate(pending):
                        vector = vectors[position] if vectors is not None else None
                        if vector is not None:
                            normalized, blob = _normalize_vector(vector)
                            dim = len(normalized)
                            model = model_name
                        else:
                            blob = None
                            dim = None
                            model = None
                        await self._delete_chunk(database, record.id, index)
                        await database.execute(
                            "INSERT OR REPLACE INTO memory_chunks ("
                            "memory_id, chunk_index, revision, title, summary, "
                            "text, text_sha256, embedding_model, embedding_dim, "
                            "embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                record.id,
                                index,
                                record.revision,
                                record.title,
                                record.summary,
                                text,
                                digest,
                                model,
                                dim,
                                blob,
                            ),
                        )
                        if self._bm25_available:
                            await database.execute(
                                "INSERT INTO memory_text ("
                                "text, memory_id, chunk_index) VALUES (?, ?, ?)",
                                (text, record.id, index),
                            )
                await database.commit()
        except Exception as exc:
            # 增量同步失败只降级检索质量，绝不破坏 Markdown 写入路径。
            logger.warning(
                "memory search index sync failed for %s: %s",
                record.id,
                exc,
            )

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        if not self._initialized:
            return MemorySearchResult(
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                query=query,
                degrade_reason="index not initialized",
            )
        
        # 两条检索路径分别保留自己的候选配额，然后取并集。这样 top_k=2
        # 时不会退化成“向量路径返回两个”，而是尽量得到 1 条向量 + 1 条 BM25。
        limit = limit or self.configs.top_k
        vector_fetch = max(
            self.configs.vector_top_k * self.configs.candidate_multiplier,
            self.configs.vector_top_k,
        )
        text_fetch = max(
            self.configs.text_top_k * self.configs.candidate_multiplier,
            self.configs.text_top_k,
        )
        vector_hits = await self._vector_search(query, vector_fetch)
        degrade_reason = "embedding unavailable or failed" if vector_hits is None else None
        text_hits = (
            None
            if self.configs.vector_only
            else await self._text_search(query, text_fetch)
        )
        if text_hits is None and vector_hits is None:
            return MemorySearchResult(
                mode=SearchMode.UNAVAILABLE,
                candidates=(),
                query=query,
                degrade_reason=degrade_reason or "no retrieval path available",
            )
        vector_memory_hits = _hit_per_memory(vector_hits or ())
        text_memory_hits = _hit_per_memory(text_hits or ())
        vector_rank_by_memory = {
            hit.memory_id: rank
            for rank, hit in enumerate(vector_memory_hits, start=1)
        }
        text_rank_by_memory = {
            hit.memory_id: rank
            for rank, hit in enumerate(text_memory_hits, start=1)
        }
        selected_ids = set(
            list(vector_rank_by_memory)[: self.configs.vector_top_k]
            + list(text_rank_by_memory)[: self.configs.text_top_k]
        )
        scores: dict[str, float] = {}
        best_chunk: dict[str, tuple[float, str, str]] = {} #多个chunk可能对应同一条memory，对于该memory，保留分数最高的chunk
        vector_scores: dict[str, float] = {}

        matched_vector: set[str] = set()
        matched_text: set[str] = set()
        for rank, hit in enumerate(vector_memory_hits):
            scores[hit.memory_id] = scores.get(hit.memory_id,0) + 1 / (_RRF + rank)
            matched_vector.add(hit.memory_id)
            vector_scores[hit.memory_id] = max(
                vector_scores.get(hit.memory_id, float("-inf")),
                hit.score,
            )
            _keep_best_chunk(best_chunk,hit)
        
        for rank, hit in enumerate(text_memory_hits):
            scores[hit.memory_id] = scores.get(hit.memory_id,0) + 1 / (_RRF + rank)
            matched_text.add(hit.memory_id)
            _keep_best_chunk(best_chunk,hit)
        
        ordered = sorted(
            selected_ids,
            key=lambda memory_id: (-scores[memory_id], memory_id),
        )
        candidates = tuple(
            MemorySearchCandidate(
                memory_id=memory_id,
                title=best_chunk[memory_id][1],
                summary="",
                revision=0,
                snippet=best_chunk[memory_id][2][: self.configs.snippet_chars],
                rrf_score=scores[memory_id],
                vector_similarity=vector_scores.get(memory_id),
                matched_by_vector=memory_id in matched_vector,
                matched_by_text=memory_id in matched_text,
                vector_rank=vector_rank_by_memory.get(memory_id),
                text_rank=text_rank_by_memory.get(memory_id),
            )
            for memory_id in ordered[:limit]
        )
        if vector_hits is not None and text_hits is not None:
            mode = SearchMode.HYBRID
        elif vector_hits is not None:
            mode = SearchMode.VECTOR
        else:
            mode = SearchMode.BM25
        return MemorySearchResult(
            mode=mode,
            candidates=candidates,
            query=query,
            degrade_reason=degrade_reason,
        )


    async def _vector_search(self,query:str,fetch:int) -> tuple[_ChunkHit,...]:
        try:
            query_vector = await self.embedding.embed_query(query)
        except Exception as exc:
            logger.warning("memory query embedding failed: %s", exc)
            return None
        query_vector, _ = _normalize_vector(query_vector)
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, title, text, embedding "
                    "FROM memory_chunks WHERE embedding IS NOT NULL"
                )
        except aiosqlite.Error as exc:
            logger.warning("memory vector search failed: %s", exc)
            return None

        hits: list[_ChunkHit] = []
        
        for memory_id, chunk_index, title, text, blob in rows:
            vector = _decode_embedding(blob)
            if len(vector) != len(query_vector):
                continue
            similarity = _cosine_similarity(array("f", query_vector), vector)
            if similarity < self.configs.min_vector_similarity:
                continue
            hits.append(
                _ChunkHit(
                    memory_id=memory_id,
                    chunk_index=chunk_index,
                    title=title or "",
                    text=text or "",
                    score=similarity,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.memory_id, hit.chunk_index))
        return tuple(hits[:fetch])


    async def _text_search(
        self,
        query: str,
        fetch: int,
        *,
        max_terms: int = 5,
    ) -> tuple[_ChunkHit, ...]:

        terms: list[str] = []
        items = query.replace("\r", " ").replace("\n", " ").split()
        for raw in items:
            term = raw.strip().strip("\"'()*")
            if len(term) < 2:
                continue
            terms.append(term)
        if not terms:
            return None
        expressions = [f'"{term}"' for term in terms[:max_terms]]
        expressions = " OR ".join(expressions)

        if expressions is None:
            return None
        try:
            async with self._connect() as database:
                rows = await database.execute_fetchall(
                    "SELECT memory_id, chunk_index, text FROM memory_text "
                    "WHERE memory_text MATCH ? ORDER BY rank LIMIT ?",
                    (expressions, fetch),
                )
        except aiosqlite.Error as exc:
            logger.warning("memory fts search failed: %s", exc)
            return None
        return tuple(
            _ChunkHit(
                memory_id=memory_id,
                chunk_index=chunk_index,
                title="",
                text=text or "",
                score=0.0,
            )
            for memory_id, chunk_index, text in rows
        )
    @property
    def _embedding_model_name(self) -> str | None:
        if self.embedding is None:
            return None
        return self.embedding.model_name

    async def _embed_texts(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...] | None:
        if self.embedding is None:
            return None
        return await self.embedding.embed_documents(texts)

    async def _delete_chunk(
        self,
        database: aiosqlite.Connection,
        memory_id: str,
        chunk_index: int,
    ) -> None:
        await database.execute(
            "DELETE FROM memory_chunks WHERE memory_id = ? AND chunk_index = ?",
            (memory_id, chunk_index),
        )
        if self._bm25_available:
            await database.execute(
                "DELETE FROM memory_text WHERE memory_id = ? AND chunk_index = ?",
                (memory_id, chunk_index),
            )

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.database_path)

def chunk_memory_text(
        record: RegularMemoryRecord,
        *,
        settings: MemorySearchConfig,
    ) -> tuple[tuple[str, str], ...]:
        """把一条记忆切成 (chunk 文本, 正文 sha256) 列表。

        Chunk 文本带 ``title | summary`` 语义头部，让向量与 FTS 都能利用
        Recall Cue；正文按段落累积切块并保留少量重叠。
        """

        header = f"{record.title} | {record.summary}"
        paragraphs = [part.strip() for part in record.content.split("\n\n")]
        bodies: list[str] = []
        current: list[str] = []
        used = 0
        for paragraph in paragraphs:
            if not paragraph:
                continue
            addition = len(paragraph) + (1 if current else 0)
            if current and used + addition > settings.chunk_chars:
                bodies.append("\n\n".join(current))
                overlap = "\n\n".join(current)[-settings.chunk_overlap_chars :].lstrip()
                current = [overlap] if overlap else []
                used = len(overlap)
            current.append(paragraph)
            used += addition
        if current:
            bodies.append("\n\n".join(current))
        if not bodies:
            bodies = [record.content]
        bodies = bodies[: settings.max_chunks_per_memory]
        chunks: list[tuple[str, str]] = []
        for body in bodies:
            text = f"{header}\n{body}"[: max(settings.chunk_chars * 2, len(header) + 200)]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append((text, digest))
        return tuple(chunks)


def _hit_per_memory(hits: tuple[_ChunkHit, ...]) -> tuple[_ChunkHit, ...]:
        """保留每条检索路径中一个 Memory 排名最靠前的 Chunk。"""

        seen: set[str] = set()
        unique: list[_ChunkHit] = []
        for hit in hits:
            if hit.memory_id in seen:
                continue
            seen.add(hit.memory_id)
            unique.append(hit)
        return tuple(unique)

def _keep_best_chunk(best_chunk: dict[str, tuple[float, str, str]], hit: _ChunkHit,):
    current = best_chunk.get(hit.memory_id)
    if current is None or hit.score > current[0]:
        best_chunk[hit.memory_id] = (hit.score, hit.title or "", hit.text)
    
    
def _decode_embedding(blob: bytes) -> array:
    values = array("f")
    values.frombytes(blob)
    return values

def _normalize_vector(
        vector: tuple[float, ...],
    ) -> tuple[tuple[float, ...], bytes]:
        values = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(values)

        if norm > 0:
            values = values / norm

        normalized = tuple(values.tolist())
        blob = values.tobytes()
        return normalized, blob
    

def _cosine_similarity(
    left: array,
    right: array,
) -> float:
    if len(left) != len(right) or not len(left):
        return 0.0
    dot = 0.0
    for index in range(len(left)):
        dot += left[index] * right[index]
    if dot == 0.0:
        return 0.0
    return dot  # 索引与查询向量均已归一化，点积即余弦。
