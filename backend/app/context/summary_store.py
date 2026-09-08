"""滚动会话摘要的 SQLite 持久化。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / ".databse" / "sidekick.db"

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
    ) -> ConversationSummaryState | None:
        """读取会话当前生效的滚动摘要。"""
        if conversation_id is None:
            raise KeyError("查询摘要的conversation_id不能为空")
        
        async with self._connect() as db:
            async with db.execute(
                "SELECT summary_json, covered_message_count FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationSummaryState(
            summary=RollingConversationSummary.model_validate_json(row[0]),
            covered_message_count=row[1],
        )

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
        async with self._connect() as db:
            cursor = await db.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            yield db


async def main() -> None:
    store = SQLiteConversationSummaryStore(DEFAULT_DATABASE_PATH.with_name("summary_test.db"))
    await store.initialize()
    conversation_id = "test"

    # 摘要需要关联一条会话记录。
    async with aiosqlite.connect(store.database_path) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY)")
        await db.execute("INSERT OR IGNORE INTO conversations (id) VALUES (?)", (conversation_id,))
        await db.commit()

    state = ConversationSummaryState(
        summary=RollingConversationSummary(current_objective="完成模型接入"),
        covered_message_count=2,
    )
    await store.save(conversation_id, state)
    result = await store.load(conversation_id)
    print(f"数据库：{store.database_path}")
    print(result.model_dump_json(indent=2) if result is not None else "未找到摘要")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
