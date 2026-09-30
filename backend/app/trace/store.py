"""基于 SQLite 的 Agent 运行轨迹存储。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite
from app.agent.events import AgentEvent, AgentEventHandler, AgentEventType
from app.conversation.store import DEFAULT_DATABASE_PATH

from .models import AgentRunTrace, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    provider TEXT,
    model TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, sequence),
    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at
ON agent_runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
ON agent_runs(conversation_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_run_sequence
ON agent_events(run_id, sequence);
"""


class TraceStore:
    """将 Agent Run 摘要和完整事件保存在 SQLite 中。"""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        """创建数据库目录和 Trace 数据表。"""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def record_event(self, event: AgentEvent) -> None:
        """幂等保存事件并更新对应 Run 摘要。"""

        event_time = event.event_time.isoformat()
        async with self._connect() as database:
            await database.execute(
                """
                INSERT OR IGNORE INTO agent_runs (
                    run_id,
                    conversation_id,
                    status,
                    started_at,
                    provider,
                    model
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.conversation_id,
                    RunStatus.RUNNING.value,
                    event_time,
                    event.provider,
                    event.model,
                ),
            )
            await database.execute(
                """
                INSERT OR IGNORE INTO agent_events (
                    event_id,
                    run_id,
                    sequence,
                    type,
                    event_time,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.sequence,
                    event.type.value,
                    event_time,
                    # exclude_none：只写有值字段，避免新增模型字段的 null 冗余
                    # 写进历史 payload，减少未来字段增删的版本漂移。
                    event.model_dump_json(exclude_none=True),
                ),
            )
            await self._update_run(database, event)
            await database.commit()

    async def get(self, run_id: str) -> AgentRunTrace | None:
        """按完整 Run ID 获取运行摘要。"""

        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        return _trace_from_row(row) if row is not None else None

    async def resolve(self, identifier: str) -> AgentRunTrace | None:
        """使用完整 ID 或唯一 ID 前缀查找 Run。"""

        normalized = identifier.strip()
        if not normalized:
            return None
        exact = await self.get(normalized)
        if exact is not None:
            return exact

        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM agent_runs
                WHERE run_id LIKE ?
                ORDER BY started_at DESC
                LIMIT 2
                """,
                (f"{normalized}%",),
            )
            rows = await cursor.fetchall()
        if len(rows) > 1:
            raise ValueError(f"Run ID 前缀不唯一：{identifier}")
        return _trace_from_row(rows[0]) if rows else None

    async def list_runs(
        self,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> tuple[AgentRunTrace, ...]:
        """按开始时间倒序列出运行记录。"""

        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = "SELECT * FROM agent_runs"
        parameters: tuple[object, ...]
        if conversation_id is None:
            parameters = (limit,)
        else:
            query += " WHERE conversation_id = ?"
            parameters = (conversation_id, limit)
        query += " ORDER BY started_at DESC LIMIT ?"

        async with self._connect() as database:
            cursor = await database.execute(query, parameters)
            rows = await cursor.fetchall()
        return tuple(_trace_from_row(row) for row in rows)

    async def load_events(self, run_id: str) -> tuple[AgentEvent, ...]:
        """按 sequence 读取一次 Run 的完整事件。"""

        if await self.get(run_id) is None:
            raise KeyError(f"Run 不存在：{run_id}")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT payload_json FROM agent_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return tuple(
            AgentEvent.model_validate_json(row["payload_json"])
            for row in rows
        )

    async def delete(self, run_id: str) -> bool:
        """删除 Run 及其全部事件。"""

        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM agent_runs WHERE run_id = ?",
                (run_id,),
            )
            await database.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def _update_run(
        database: aiosqlite.Connection,
        event: AgentEvent,
    ) -> None:
        status: RunStatus | None = None
        completed_at: str | None = None
        if event.type is AgentEventType.AGENT_COMPLETED:
            status = RunStatus.COMPLETED
            completed_at = event.event_time.isoformat()
        elif event.type is AgentEventType.AGENT_FAILED:
            status = RunStatus.FAILED
            completed_at = event.event_time.isoformat()

        updates_main_model = event.type in {
            AgentEventType.AGENT_STARTED,
            AgentEventType.MODEL_STARTED,
            AgentEventType.MODEL_COMPLETED,
            AgentEventType.AGENT_COMPLETED,
            AgentEventType.AGENT_FAILED,
        }
        provider = event.provider if updates_main_model else None
        model = event.model if updates_main_model else None
        usage = event.usage if updates_main_model else None
        await database.execute(
            """
            UPDATE agent_runs
            SET
                conversation_id = COALESCE(?, conversation_id),
                status = COALESCE(?, status),
                completed_at = COALESCE(?, completed_at),
                provider = COALESCE(?, provider),
                model = COALESCE(?, model),
                steps = MAX(steps, ?),
                stop_reason = COALESCE(?, stop_reason),
                input_tokens = MAX(input_tokens, ?),
                output_tokens = MAX(output_tokens, ?),
                total_tokens = MAX(total_tokens, ?),
                event_count = (
                    SELECT COUNT(*) FROM agent_events WHERE run_id = ?
                )
            WHERE run_id = ?
            """,
            (
                event.conversation_id,
                status.value if status else None,
                completed_at,
                provider,
                model,
                event.step or 0,
                event.stop_reason.value if event.stop_reason else None,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0,
                usage.total_tokens if usage else 0,
                event.run_id,
                event.run_id,
            ),
        )

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        await database.execute("PRAGMA foreign_keys = ON")
        try:
            yield database
        finally:
            await database.close()


class TraceEventHandler(AgentEventHandler):
    """把 Runtime 事件逐条写入 SQLite TraceStore。"""

    def __init__(self, store: TraceStore) -> None:
        self.store = store

    async def handle(self, event: AgentEvent) -> None:
        # 持久化每个增量 chunk 会放大 Trace 与数据库。
        if event.type in (
            AgentEventType.MODEL_OUTPUT_DELTA,
            AgentEventType.MODEL_REASONING_DELTA,
        ):
            return
        await self.store.record_event(event)


def _trace_from_row(row: aiosqlite.Row) -> AgentRunTrace:
    return AgentRunTrace(
        run_id=row["run_id"],
        conversation_id=row["conversation_id"],
        status=row["status"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None
        ),
        provider=row["provider"],
        model=row["model"],
        steps=row["steps"],
        stop_reason=row["stop_reason"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        event_count=row["event_count"],
    )
