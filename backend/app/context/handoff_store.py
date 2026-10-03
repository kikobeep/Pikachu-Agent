"""Handoff 快照的文件式持久化。

沿用 PlanStore 的原子写入模式：tmp 文件 → fsync → os.replace。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from app.context.handoff import HandoffSnapshot

DEFAULT_HANDOFFS_DIR = Path(__file__).resolve().parents[2] / ".database" / "handoffs"


class HandoffStore:
    """每个 conversation_id 对应一个 handoff_<id>.json。

    多次 handoff 只保留最新一份（同路径覆盖），因为每次都是全量快照。
    """

    def __init__(self, handoffs_dir: str | Path = DEFAULT_HANDOFFS_DIR) -> None:
        self.handoffs_dir = Path(handoffs_dir).expanduser().resolve()
        self._locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self.handoffs_dir.mkdir, parents=True, exist_ok=True
        )

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    def _path_for(self, conversation_id: str) -> Path:
        return self.handoffs_dir / f"handoff_{conversation_id}.json"

    async def save(self, snapshot: HandoffSnapshot) -> None:
        async with self._lock_for(snapshot.conversation_id):
            await asyncio.to_thread(
                _write_snapshot,
                self._path_for(snapshot.conversation_id),
                snapshot,
            )

    async def load(self, conversation_id: str) -> HandoffSnapshot | None:
        async with self._lock_for(conversation_id):
            return await asyncio.to_thread(
                _read_snapshot, self._path_for(conversation_id)
            )

    async def clear(self, conversation_id: str) -> None:
        async with self._lock_for(conversation_id):
            await asyncio.to_thread(
                _delete_snapshot, self._path_for(conversation_id)
            )


def _write_snapshot(path: Path, snapshot: HandoffSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_snapshot(path: Path) -> HandoffSnapshot | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return HandoffSnapshot.model_validate(data)


def _delete_snapshot(path: Path) -> None:
    if path.exists():
        path.unlink()
