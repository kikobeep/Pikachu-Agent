"""持久化外部评测器对 Run 的结果。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import aiosqlite


class RunOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_outcomes (
    run_id TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    exercise TEXT NOT NULL,
    phase TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('passed', 'failed')),
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


class RunOutcomeStore:
    """保存评测结果；Run 本身只记录 Agent 生命周期。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as database:
            await database.executescript(_SCHEMA)
            await database.commit()

    async def save(
        self,
        *,
        run_id: str,
        language: str,
        exercise: str,
        phase: str,
        outcome: RunOutcome,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.database_path) as database:
            await database.execute(
                """
                INSERT INTO run_outcomes (
                    run_id, language, exercise, phase, outcome,
                    stdout, stderr, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    language = excluded.language,
                    exercise = excluded.exercise,
                    phase = excluded.phase,
                    outcome = excluded.outcome,
                    stdout = excluded.stdout,
                    stderr = excluded.stderr,
                    created_at = excluded.created_at
                """,
                (
                    run_id,
                    language,
                    exercise,
                    phase,
                    outcome.value,
                    stdout[-20_000:],
                    stderr[-20_000:],
                    now,
                ),
            )
            await database.commit()

    async def get(self, run_id: str) -> RunOutcome | None:
        async with aiosqlite.connect(self.database_path) as database:
            cursor = await database.execute(
                "SELECT outcome FROM run_outcomes WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        return RunOutcome(row[0]) if row is not None else None


__all__ = ["RunOutcome", "RunOutcomeStore"]
