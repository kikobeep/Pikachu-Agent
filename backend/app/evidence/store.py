"""基于 SQLite 的不可变 Evidence Store。"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.conversation.store import DEFAULT_DATABASE_PATH

from .models import EvidenceDocument, EvidenceRecord, EvidenceSearchHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_chars INTEGER NOT NULL,
    content_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    task_id TEXT,
    task_step_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, tool_call_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_conversation_created
ON evidence(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_conversation_task
ON evidence(conversation_id, task_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_run
ON evidence(run_id, created_at ASC);
"""
_EVIDENCE_ID_RE = re.compile(r"^[0-9a-f]{4,32}$")
DEFAULT_MAX_EVIDENCE_ITEM_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_EVIDENCE_TOTAL_BYTES = 512 * 1024 * 1024


class EvidenceCapacityError(ValueError):
    """Evidence 容量边界拒绝了新原文，已有证据保持不变。"""


class EvidenceStore:
    """保存完整工具输出，并强制所有模型查询先按会话隔离。"""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        max_item_bytes: int = DEFAULT_MAX_EVIDENCE_ITEM_BYTES,
        max_total_bytes: int = DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
    ) -> None:
        if max_item_bytes < 1 or max_total_bytes < 1:
            raise ValueError("evidence capacity limits must be positive")
        if max_item_bytes > max_total_bytes:
            raise ValueError("max_item_bytes cannot exceed max_total_bytes")
        self.database_path = Path(database_path).expanduser().resolve()
        self.max_item_bytes = max_item_bytes
        self.max_total_bytes = max_total_bytes

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def create(
        self,
        *,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        content: str,
        sha256: str,
        task_id: str | None = None,
        task_step_id: str | None = None,
    ) -> EvidenceRecord:
        """创建不可变证据；同一 Run/ToolCall 的相同内容幂等返回。"""

        conversation_id = _required(conversation_id, "conversation_id")
        run_id = _required(run_id, "run_id")
        tool_call_id = _required(tool_call_id, "tool_call_id")
        tool_name = _required(tool_name, "tool_name")
        evidence_id = uuid4().hex
        created_at = datetime.now(UTC)
        content_bytes = len(content.encode("utf-8"))
        async with self._connect() as database:
            try:
                # 串行化“幂等检查 + 容量检查 + 写入”，避免并发重放绕过限制。
                await database.execute("BEGIN IMMEDIATE")
                cursor = await database.execute(
                    "SELECT * FROM evidence WHERE run_id = ? AND tool_call_id = ?",
                    (run_id, tool_call_id),
                )
                existing = await cursor.fetchone()
                if existing is not None:
                    if (
                        existing["sha256"] != sha256
                        or existing["conversation_id"] != conversation_id
                        or existing["tool_name"] != tool_name
                    ):
                        raise ValueError(
                            "evidence conflict: tool call already has different facts"
                        )
                    await database.commit()
                    return _record_from_row(existing)
                if content_bytes > self.max_item_bytes:
                    raise EvidenceCapacityError(
                        "evidence item exceeds max_item_bytes: "
                        f"{content_bytes} > {self.max_item_bytes}"
                    )
                cursor = await database.execute(
                    "SELECT COALESCE(SUM(content_bytes), 0) AS total FROM evidence"
                )
                total_bytes = int((await cursor.fetchone())["total"])
                if total_bytes + content_bytes > self.max_total_bytes:
                    raise EvidenceCapacityError(
                        "evidence store exceeds max_total_bytes: "
                        f"{total_bytes + content_bytes} > {self.max_total_bytes}"
                    )
                await database.execute(
                    """
                    INSERT INTO evidence (
                        id, conversation_id, run_id, tool_call_id, tool_name,
                        content_type, content, content_chars, content_bytes, sha256,
                        task_id, task_step_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        conversation_id,
                        run_id,
                        tool_call_id,
                        tool_name,
                        "text/plain; charset=utf-8",
                        content,
                        len(content),
                        content_bytes,
                        sha256,
                        task_id,
                        task_step_id,
                        created_at.isoformat(),
                    ),
                )
                await database.commit()
            except Exception:
                await database.rollback()
                raise
        return EvidenceRecord(
            id=evidence_id,
            conversation_id=conversation_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content_chars=len(content),
            content_bytes=content_bytes,
            sha256=sha256,
            task_id=task_id,
            task_step_id=task_step_id,
            created_at=created_at,
        )

    async def resolve(
        self,
        identifier: str,
        *,
        conversation_id: str,
    ) -> EvidenceDocument | None:
        """在当前会话内按完整 ID 或唯一前缀读取证据。"""

        normalized = identifier.strip().lower()
        if not _EVIDENCE_ID_RE.fullmatch(normalized):
            raise ValueError(
                "evidence id must be a 4-32 character hexadecimal prefix"
            )
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence
                WHERE conversation_id = ? AND id LIKE ?
                ORDER BY created_at DESC LIMIT 2
                """,
                (_required(conversation_id, "conversation_id"), f"{normalized}%"),
            )
            rows = await cursor.fetchall()
        if len(rows) > 1:
            raise ValueError(f"Evidence ID 前缀不唯一：{identifier}")
        if not rows:
            return None
        return EvidenceDocument(
            record=_record_from_row(rows[0]),
            content=rows[0]["content"],
        )

    async def search(
        self,
        *,
        conversation_id: str,
        query: str,
        tool_name: str | None = None,
        task_id: str | None = None,
        limit: int = 10,
    ) -> tuple[EvidenceSearchHit, ...]:
        """在当前会话内检索原始输出，返回有界片段而非整份内容。"""

        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string")
        filters = ["conversation_id = ?", "instr(lower(content), lower(?)) > 0"]
        parameters: list[object] = [
            _required(conversation_id, "conversation_id"),
            normalized_query,
        ]
        if tool_name:
            filters.append("tool_name = ?")
            parameters.append(tool_name.strip())
        if task_id:
            filters.append("task_id = ?")
            parameters.append(task_id.strip())
        parameters.append(limit)
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM evidence WHERE "
                + " AND ".join(filters)
                + " ORDER BY created_at DESC LIMIT ?",
                parameters,
            )
            rows = await cursor.fetchall()
        return tuple(
            EvidenceSearchHit(
                record=_record_from_row(row),
                snippet=_snippet(row["content"], normalized_query),
            )
            for row in rows
        )

    async def list_recent(
        self,
        *,
        conversation_id: str,
        limit: int = 12,
    ) -> tuple[EvidenceRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence WHERE conversation_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (_required(conversation_id, "conversation_id"), limit),
            )
            rows = await cursor.fetchall()
        return tuple(_record_from_row(row) for row in rows)

    async def list_for_task(
        self,
        *,
        conversation_id: str,
        task_id: str,
        limit: int = 20,
    ) -> tuple[EvidenceRecord, ...]:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM evidence
                WHERE conversation_id = ? AND task_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (
                    _required(conversation_id, "conversation_id"),
                    _required(task_id, "task_id"),
                    limit,
                ),
            )
            rows = await cursor.fetchall()
        return tuple(_record_from_row(row) for row in rows)

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


def _record_from_row(row: aiosqlite.Row) -> EvidenceRecord:
    return EvidenceRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        run_id=row["run_id"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        content_type=row["content_type"],
        content_chars=row["content_chars"],
        content_bytes=row["content_bytes"],
        sha256=row["sha256"],
        task_id=row["task_id"],
        task_step_id=row["task_step_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _snippet(content: str, query: str, *, radius: int = 180) -> str:
    index = content.casefold().find(query.casefold())
    if index < 0:
        return content[: radius * 2]
    start = max(0, index - radius)
    end = min(len(content), index + len(query) + radius)
    prefix = "…" if start else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


def _required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


__all__ = [
    "DEFAULT_MAX_EVIDENCE_ITEM_BYTES",
    "DEFAULT_MAX_EVIDENCE_TOTAL_BYTES",
    "EvidenceCapacityError",
    "EvidenceStore",
]
