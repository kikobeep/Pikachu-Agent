"""Light 模式的会话级状态仓库。

状态严格按 conversation_id 隔离。它保存结构化状态和关系，不保存完整对话；
原始工具输出仍由既有 EvidenceStore 负责保存。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.conversation.store import DEFAULT_DATABASE_PATH

from .models import (
    LightStateDelta,
    LightStateNode,
    LightStateOperation,
    LightStateRelation,
    LightStateType,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS light_state_nodes (
    state_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    state_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    value_json TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    source_run_id TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_light_state_nodes_conversation
ON light_state_nodes(conversation_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS light_state_relations (
    conversation_id TEXT NOT NULL,
    source_state_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_state_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, source_state_id, relation, target_state_id)
);

CREATE INDEX IF NOT EXISTS idx_light_state_relations_target
ON light_state_relations(conversation_id, target_state_id);
"""

_ALLOWED_RELATIONS = frozenset(
    {"derives", "supports", "follows", "supersedes"}
)


class LightStateStore:
    """持久化 FlowState 节点；所有查询都显式要求 conversation_id。"""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def historical_index(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> tuple[LightStateNode, ...]:
        """返回当前会话的轻量索引；不跨 conversation_id 读取。"""

        conversation_id = _required(conversation_id, "conversation_id")
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM light_state_nodes
                WHERE conversation_id = ? AND status = 'active'
                ORDER BY updated_at DESC LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = await cursor.fetchall()
            return await _hydrate_nodes(database, conversation_id, rows)

    async def active_for_run(
        self,
        conversation_id: str,
        run_id: str,
        focused_ids: tuple[str, ...] = (),
        recovered_run_id: str | None = None,
    ) -> tuple[LightStateNode, ...]:
        """返回当前 Run 新写入的状态，以及模型主动 focus 的历史状态。"""

        conversation_id = _required(conversation_id, "conversation_id")
        run_id = _required(run_id, "run_id")
        ids = tuple(dict.fromkeys(focused_ids))
        run_ids = tuple(
            dict.fromkeys(
                (
                    run_id,
                    *(() if recovered_run_id is None else (recovered_run_id,)),
                )
            )
        )
        run_ids = tuple(value for value in run_ids if value)
        clauses = ["conversation_id = ?", "status = 'active'"]
        parameters: list[object] = [conversation_id]
        run_placeholders = ",".join("?" for _ in run_ids)
        if ids:
            state_placeholders = ",".join("?" for _ in ids)
            clauses.append(
                f"(source_run_id IN ({run_placeholders}) OR "
                f"state_id IN ({state_placeholders}))"
            )
            parameters.extend((*run_ids, *ids))
        else:
            clauses.append(f"source_run_id IN ({run_placeholders})")
            parameters.extend(run_ids)
        async with self._connect() as database:
            cursor = await database.execute(
                f"SELECT * FROM light_state_nodes WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at DESC",
                parameters,
            )
            rows = await cursor.fetchall()
            return await _hydrate_nodes(database, conversation_id, rows)

    async def get(
        self,
        conversation_id: str,
        state_id: str,
    ) -> LightStateNode | None:
        conversation_id = _required(conversation_id, "conversation_id")
        state_id = _required(state_id, "state_id")
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM light_state_nodes
                WHERE conversation_id = ? AND state_id = ? AND status = 'active'
                """,
                (conversation_id, state_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            nodes = await _hydrate_nodes(database, conversation_id, (row,))
        return nodes[0]

    async def apply_deltas(
        self,
        conversation_id: str,
        run_id: str,
        deltas: tuple[LightStateDelta, ...],
    ) -> tuple[LightStateNode, ...]:
        """原子应用一批 state_delta，并校验关系不能越过当前会话。"""

        conversation_id = _required(conversation_id, "conversation_id")
        run_id = _required(run_id, "run_id")
        if not deltas:
            return ()
        async with self._connect() as database:
            await database.execute("BEGIN IMMEDIATE")
            changed: list[LightStateNode] = []
            try:
                for delta in deltas:
                    node = await _apply_delta(
                        database,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        delta=delta,
                    )
                    if node is not None:
                        changed.append(node)
                await database.commit()
            except Exception:
                await database.rollback()
                raise
        return tuple(changed)

    @asynccontextmanager
    async def _connect(self):
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


async def _apply_delta(
    database: aiosqlite.Connection,
    *,
    conversation_id: str,
    run_id: str,
    delta: LightStateDelta,
) -> LightStateNode | None:
    now = _now()
    if delta.operation is LightStateOperation.ADD:
        state_id = delta.state_id or f"S{uuid4().hex[:12]}"
        if not delta.state_type or not (delta.summary or "").strip():
            raise ValueError("add state_delta requires state_type and summary")
        await database.execute(
            """
            INSERT INTO light_state_nodes (
                state_id, conversation_id, state_type, summary, value_json,
                evidence_ids_json, source_run_id, revision, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'active', ?, ?)
            """,
            (
                state_id,
                conversation_id,
                delta.state_type.value,
                delta.summary.strip(),
                json.dumps(delta.value or {}, ensure_ascii=False, sort_keys=True),
                json.dumps(delta.evidence_ids, ensure_ascii=False),
                run_id,
                now,
                now,
            ),
        )
    else:
        if not delta.state_id:
            raise ValueError(f"{delta.operation.value} state_delta requires state_id")
        row = await _require_node(database, conversation_id, delta.state_id)
        if delta.operation is LightStateOperation.REMOVE:
            await database.execute(
                "UPDATE light_state_nodes SET status = 'removed', revision = revision + 1, updated_at = ? "
                "WHERE conversation_id = ? AND state_id = ?",
                (now, conversation_id, delta.state_id),
            )
        else:
            state_type = delta.state_type or LightStateType(row["state_type"])
            summary = (delta.summary or row["summary"]).strip()
            value = delta.value if delta.value is not None else json.loads(row["value_json"])
            evidence_ids = delta.evidence_ids or tuple(json.loads(row["evidence_ids_json"]))
            await database.execute(
                """
                UPDATE light_state_nodes
                SET state_type = ?, summary = ?, value_json = ?, evidence_ids_json = ?,
                    source_run_id = ?, revision = revision + 1, updated_at = ?
                WHERE conversation_id = ? AND state_id = ?
                """,
                (
                    state_type.value,
                    summary,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    json.dumps(evidence_ids, ensure_ascii=False),
                    run_id,
                    now,
                    conversation_id,
                    delta.state_id,
                ),
            )
        if delta.operation is LightStateOperation.REMOVE:
            return None
        state_id = delta.state_id

    node = await _require_node(database, conversation_id, state_id)
    for relation in delta.relations:
        if relation.relation not in _ALLOWED_RELATIONS:
            raise ValueError(f"unsupported state relation: {relation.relation}")
        target = await _require_node(database, conversation_id, relation.target_state_id)
        if target["status"] != "active":
            raise ValueError("state relation target must be active")
        await database.execute(
            """
            INSERT OR IGNORE INTO light_state_relations (
                conversation_id, source_state_id, relation, target_state_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                state_id,
                relation.relation,
                relation.target_state_id,
                now,
            ),
        )
    return _node_from_row(node)


async def _require_node(
    database: aiosqlite.Connection,
    conversation_id: str,
    state_id: str,
):
    cursor = await database.execute(
        "SELECT * FROM light_state_nodes WHERE conversation_id = ? AND state_id = ?",
        (conversation_id, state_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(
            f"state_id {state_id!r} does not exist in conversation {conversation_id!r}"
        )
    return row


def _node_from_row(row: aiosqlite.Row) -> LightStateNode:
    return LightStateNode(
        state_id=row["state_id"],
        conversation_id=row["conversation_id"],
        state_type=LightStateType(row["state_type"]),
        summary=row["summary"],
        value=json.loads(row["value_json"]),
        evidence_ids=tuple(json.loads(row["evidence_ids_json"])),
        source_run_id=row["source_run_id"],
        revision=row["revision"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _hydrate_nodes(
    database: aiosqlite.Connection,
    conversation_id: str,
    rows,
) -> tuple[LightStateNode, ...]:
    nodes = tuple(_node_from_row(row) for row in rows)
    if not nodes:
        return ()
    ids = tuple(node.state_id for node in nodes)
    placeholders = ",".join("?" for _ in ids)
    cursor = await database.execute(
        f"""
        SELECT source_state_id, relation, target_state_id
        FROM light_state_relations
        WHERE conversation_id = ? AND source_state_id IN ({placeholders})
        """,
        (conversation_id, *ids),
    )
    relations_by_source: dict[str, list[LightStateRelation]] = {}
    for row in await cursor.fetchall():
        relations_by_source.setdefault(row["source_state_id"], []).append(
            LightStateRelation(
                relation=row["relation"],
                target_state_id=row["target_state_id"],
            )
        )
    return tuple(
        node.model_copy(
            update={"relations": tuple(relations_by_source.get(node.state_id, ())) }
        )
        for node in nodes
    )


def _required(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["LightStateStore"]
