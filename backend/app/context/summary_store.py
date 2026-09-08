"""滚动会话摘要的 SQLite 持久化。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from app.conversation import DEFAULT_DATABASE_PATH

from .summary import ConversationSummaryState, RollingConversationSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    covered_message_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


class SQLiteConversationSummaryStore:
    """保存模型请求使用的摘要缓存，不替代完整消息历史。"""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        """创建摘要表"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()


    async def load(
        self,
        conversation_id: str,
    ) -> dict[str,Any] | None:
        """读取会话当前生效的滚动摘要。"""
        if conversation_id is None:
            raise KeyError("查询摘要的conversation_id不能为空")
        
        async with aiosqlite.connect(self.database_path) as db:
             row = await database.execute(
                """
                SELECT summary_json, covered_message_count
                FROM conversation_summaries
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return {
                "summary": RollingConversationSummary.model_validate_json(row[0]),
                "covered_message_count": row[1],
            }
        

    async def save(
        self,
        conversation_id: str,
        state: ConversationSummaryState,
    ) -> None:
        """新增或覆盖会话摘要。"""

        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO conversation_summaries (
                    conversation_id,
                    summary_json,
                    covered_message_count,
                    updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    covered_message_count = excluded.covered_message_count,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    state.summary.model_dump_json(),
                    state.covered_message_count,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await database.commit()

    async def delete(self, conversation_id: str) -> bool:
        

    


__all__ = ["SQLiteConversationSummaryStore"]
