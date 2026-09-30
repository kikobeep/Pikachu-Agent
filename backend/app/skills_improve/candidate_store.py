"""SQLite 持久化 Skill Candidate 与审核流转。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import yaml

from app.skills.config import SKILL_DESCRIPTION_MAX_LENGTH
from app.skills.utils import validate_skill_name

from .models import CandidateStatus, SkillCandidate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_candidates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    source_run_ids TEXT NOT NULL,
    cluster_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_skill_candidates_status
ON skill_candidates(status, created_at DESC);
"""


class SkillCandidateStore:
    """保存候选并处理「人工审核」的 accept / reject 流转。

    ``accept`` 会把候选渲染成 SKILL.md 写到 ``skill_project_dir``（创建或覆盖同名
    正式 Skill），再标记为 accepted；``reject`` 只标记状态，不产生任何文件。
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        skill_project_dir: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.skill_project_dir = (
            Path(skill_project_dir).expanduser().resolve()
            if skill_project_dir is not None
            else None
        )

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def create(self, candidate: SkillCandidate) -> SkillCandidate:
        """写入一条 pending 候选；同 id 重复写入幂等返回。"""

        async with self._connect() as database:
            await database.execute(
                """
                INSERT INTO skill_candidates (
                    id, name, description, content, source_run_ids,
                    cluster_id, status, created_at, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.name,
                    candidate.description,
                    candidate.content,
                    json.dumps(list(candidate.source_run_ids), ensure_ascii=False),
                    candidate.cluster_id,
                    candidate.status.value,
                    candidate.created_at.isoformat(),
                    candidate.reviewed_at.isoformat()
                    if candidate.reviewed_at is not None
                    else None,
                ),
            )
            await database.commit()
        return candidate

    async def get(self, candidate_id: str) -> SkillCandidate | None:
        async with self._connect() as database:
            cursor = await database.execute(
                "SELECT * FROM skill_candidates WHERE id = ?", (candidate_id,)
            )
            row = await cursor.fetchone()
        return _candidate_from_row(row) if row is not None else None

    async def list_by_status(
        self,
        status: CandidateStatus,
    ) -> tuple[SkillCandidate, ...]:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT * FROM skill_candidates WHERE status = ?
                ORDER BY created_at DESC
                """,
                (status.value,),
            )
            rows = await cursor.fetchall()
        return tuple(_candidate_from_row(row) for row in rows)

    async def list_pending(self) -> tuple[SkillCandidate, ...]:
        return await self.list_by_status(CandidateStatus.PENDING)

    async def accept(self, candidate_id: str) -> SkillCandidate:
        """审核通过：落盘正式 Skill 并标记 accepted。"""

        candidate = await self.get(candidate_id)
        if candidate is None:
            raise KeyError(f"candidate 不存在：{candidate_id}")
        if candidate.status is not CandidateStatus.PENDING:
            raise ValueError(f"candidate 已审核：{candidate.status.value}")

        await asyncio.to_thread(self._write_skill, candidate)
        reviewed_at = datetime.now(UTC)
        async with self._connect() as database:
            await database.execute(
                """
                UPDATE skill_candidates
                SET status = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (CandidateStatus.ACCEPTED.value, reviewed_at.isoformat(), candidate_id),
            )
            await database.commit()
        return candidate.model_copy(
            update={"status": CandidateStatus.ACCEPTED, "reviewed_at": reviewed_at}
        )

    async def reject(self, candidate_id: str) -> SkillCandidate:
        """审核拒绝：仅标记状态，不落盘任何文件。"""

        candidate = await self.get(candidate_id)
        if candidate is None:
            raise KeyError(f"candidate 不存在：{candidate_id}")
        if candidate.status is not CandidateStatus.PENDING:
            raise ValueError(f"candidate 已审核：{candidate.status.value}")
        reviewed_at = datetime.now(UTC)
        async with self._connect() as database:
            await database.execute(
                """
                UPDATE skill_candidates
                SET status = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (CandidateStatus.REJECTED.value, reviewed_at.isoformat(), candidate_id),
            )
            await database.commit()
        return candidate.model_copy(
            update={"status": CandidateStatus.REJECTED, "reviewed_at": reviewed_at}
        )

    def _write_skill(self, candidate: SkillCandidate) -> None:
        if self.skill_project_dir is None:
            raise ValueError("skill_project_dir 未配置，无法落盘正式 Skill")
        validate_skill_name(candidate.name)
        if len(candidate.description) > SKILL_DESCRIPTION_MAX_LENGTH:
            raise ValueError(
                f"description 超过 {SKILL_DESCRIPTION_MAX_LENGTH} 字符"
            )
        skill_dir = self.skill_project_dir / candidate.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        front = yaml.safe_dump(
            {"name": candidate.name, "description": candidate.description},
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        document = f"---\n{front}\n---\n\n{candidate.content.strip()}\n"
        (skill_dir / "SKILL.md").write_text(document, encoding="utf-8")

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        database = await aiosqlite.connect(self.database_path)
        database.row_factory = aiosqlite.Row
        try:
            yield database
        finally:
            await database.close()


def _candidate_from_row(row: aiosqlite.Row) -> SkillCandidate:
    return SkillCandidate(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        content=row["content"],
        source_run_ids=tuple(json.loads(row["source_run_ids"] or "[]")),
        cluster_id=row["cluster_id"],
        status=CandidateStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        reviewed_at=(
            datetime.fromisoformat(row["reviewed_at"])
            if row["reviewed_at"] is not None
            else None
        ),
    )


__all__ = ["SkillCandidateStore"]
