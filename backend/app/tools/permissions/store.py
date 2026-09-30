from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from app.conversation.store import DEFAULT_DATABASE_PATH

from .models import ApprovalScope, PermissionRule

_SCHEMA = """
CREATE TABLE IF NOT EXISTS permission_rules (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    effect TEXT NOT NULL,
    matcher_type TEXT NOT NULL,
    matcher_json TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_permission_rules_scope
ON permission_rules(scope_id);
"""


class RuleStore(ABC):
    """审批规则的存储接口。"""

    @abstractmethod
    async def add(self, rule: PermissionRule) -> None:
        """保存一条规则。"""

    @abstractmethod
    async def list(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[PermissionRule, ...]:
        """列出规则；提供 scope_ids 时只返回属于这些作用域的规则。"""

    @abstractmethod
    async def get(self, rule_id: str) -> PermissionRule | None:
        """按完整 ID 获取规则。"""

    @abstractmethod
    async def remove(self, rule_id: str) -> bool:
        """删除规则，返回是否实际删除。"""

    @abstractmethod
    async def remove_scope(
        self,
        scope: ApprovalScope,
        scope_id: str,
    ) -> int:
        """删除指定作用域的全部规则，并返回删除数量。"""

        

class PermissionRuleStore(RuleStore):
    """把规则持久化到 SQLite。"""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.execute(
                "DELETE FROM permission_rules "
                "WHERE matcher_type IN ('command_prefix', 'command_contains', "
                "'host_exact')"
            )
            await database.execute(
                "UPDATE permission_rules SET scope = 'conversation' "
                "WHERE scope = 'project'"
            )
            await database.commit()

    async def add(self, rule: PermissionRule) -> None:
        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO permission_rules (
                    id, tool_name, scope, scope_id, effect,
                    matcher_type, matcher_json, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.tool_name,
                    rule.scope.value,
                    rule.scope_id,
                    rule.effect.value,
                    rule.matcher_type,
                    json.dumps(
                        rule.matcher,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    rule.description,
                    rule.created_at.isoformat(),
                ),
            )
            await database.commit()

    async def list(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[PermissionRule, ...]:
        if scope_ids is None:
            query = "SELECT * FROM permission_rules"
            parameters: tuple[Any, ...] = ()
        elif not scope_ids:
            return ()
        else:
            placeholders = ", ".join("?" for _ in scope_ids)
            query = (
                "SELECT * FROM permission_rules "
                f"WHERE scope_id IN ({placeholders})"
            )
            parameters = scope_ids
        query += " ORDER BY created_at DESC, id DESC"
        async with self._connect() as database:
            cursor = await database.execute(query, parameters)
            rows = await cursor.fetchall()
        return tuple(_rule_from_row(row) for row in rows)

    async def get(self, rule_id: str) -> PermissionRule | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM permission_rules WHERE id = ?",
                (rule_id,),
            )
            row = await cursor.fetchone()
        return _rule_from_row(row) if row is not None else None

    async def remove(self, rule_id: str) -> bool:
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM permission_rules WHERE id = ?",
                (rule_id,),
            )
            await database.commit()
        return cursor.rowcount > 0

    async def remove_scope(self, scope: ApprovalScope, scope_id: str) -> int:
        async with self._connect() as database:
            cursor = await database.execute(
                "DELETE FROM permission_rules WHERE scope = ? AND scope_id = ?",
                (scope.value, scope_id),
            )
            await database.commit()
        return max(cursor.rowcount, 0)

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


def _rule_from_row(row: aiosqlite.Row) -> PermissionRule:
    return PermissionRule(
        id=row["id"],
        tool_name=row["tool_name"],
        scope=row["scope"],
        scope_id=row["scope_id"],
        effect=row["effect"],
        matcher_type=row["matcher_type"],
        matcher=json.loads(row["matcher_json"]),
        description=row["description"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = [
    "PermissionRuleStore"
]
