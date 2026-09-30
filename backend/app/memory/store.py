"""长期记忆的 Markdown 文件存储。

目录结构：

```text
.memory/
├── CORE.md
├── INDEX.md
├── active/M001.md ...
└── archive/M001.md
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from .utils import normalize_memory_id, write_atomic, next_memory_id,convert_to_record
from .regular import MemoryStatus, RegularMemoryRecord

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / ".memory"


logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, memory_dir: str | Path, *, max_active: int = 100) -> None:
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.active_dir = self.memory_dir / "active"
        self.archive_dir = self.memory_dir / "archive"
        self.max_active = max_active

    async def initialize(self):
        await asyncio.to_thread(self.active_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self.archive_dir.mkdir, parents=True, exist_ok=True)

    async def load(self, memory_id: str) -> RegularMemoryRecord | None:
        # 大写转换并且校验memory_id的有效性
        normalized_id = normalize_memory_id(memory_id)
        # 获取对应id的memory路径
        path = await self._get_memory_path(normalized_id)
        if path is None or not await asyncio.to_thread(path.is_file):
            return None
        # 读path，转化为memoryrecord
        record = await asyncio.to_thread(convert_to_record, path)
        return record

    async def update(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        content: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> RegularMemoryRecord:
        record = await self.load(memory_id)
        if record is None:
            raise KeyError(f"memory '{memory_id}' not found")
        if record.status is not MemoryStatus.ACTIVE:
            raise ValueError("only active memory can be updated")

        if expected_revision is not None and record.revision != expected_revision:
            raise ValueError(
                f"memory '{record.id}' revision conflict: "
                f"expected {expected_revision}, current {record.revision}"
            )
        kwargs = {
            **record.model_dump(),
            "title": title if title is not None else record.title,
            "summary": summary if summary is not None else record.summary,
            "content": content,
            "last_update_reason": reason,
            "updated_at": datetime.now(UTC),
            "revision": record.revision + 1,
        }
        update = RegularMemoryRecord(**kwargs)
        await self.write(update)
        return update

    async def write(self, record: RegularMemoryRecord):
        if record.status is not MemoryStatus.ACTIVE:
            raise ValueError("inactive memory cannot be written to active directory")
        path = self.active_dir / f"{record.id}.md"
        content = record.render_markdown()
        await asyncio.to_thread(write_atomic, path, content)

    async def list_active(self) -> tuple[RegularMemoryRecord, ...]:
        records = []
        for path in sorted(self.active_dir.glob("M*.md")):
            if await asyncio.to_thread(path.is_symlink):
                continue
            try:
                record = await asyncio.to_thread(convert_to_record, path)
                if record.status is MemoryStatus.ACTIVE:
                    records.append(record)
            except (ValueError, OSError) as exc:
                logger.warning("skip unreadable memory %s: %s", path.name, exc)
        return tuple(sorted(records, key=lambda record: record.id))

    async def count_active(self) -> int:
        return len(await self.list_active())

    async def list_archived(self) -> tuple[RegularMemoryRecord, ...]:
        """按 ID 顺序列出已归档记忆。"""

        records = []
        for path in sorted(self.archive_dir.glob("M*.md")):
            if await asyncio.to_thread(path.is_symlink):
                continue
            try:
                record = await asyncio.to_thread(convert_to_record, path)
                if record.status is MemoryStatus.ARCHIVED:
                    records.append(record)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "skip unreadable archived memory %s: %s", path.name, exc
                )
        return tuple(sorted(records, key=lambda record: record.id))

    async def archive(
        self,
        memory_id: str,
        *,
        expected_record: RegularMemoryRecord | None = None,
        reason: str | None = None,
    ) -> RegularMemoryRecord:
        record = await self.load(memory_id)
        if record is None:
            raise KeyError(f"memory '{memory_id}' not found")
        if expected_record is not None and record != expected_record:
            raise ValueError("memory changed since maintenance snapshot")
        if record.status is MemoryStatus.ARCHIVED:
            return record
        updated = RegularMemoryRecord(
            **{
                **record.model_dump(),
                "status": MemoryStatus.ARCHIVED,
                "archive_reason": reason,
                "updated_at": datetime.now(UTC),
                "revision": record.revision + 1,
            }
        )
        await self.initialize()
        source = self.active_dir / f"{record.id}.md"
        target = self.archive_dir / f"{record.id}.md"

        await asyncio.to_thread(write_atomic, source, updated.render_markdown())
        try:
            await asyncio.to_thread(os.replace, source, target)
        except BaseException:
            await asyncio.to_thread(write_atomic, source, record.render_markdown())
            raise
        return updated

    async def create(
        self,
        *,
        title: str,
        summary: str,
        content: str,
    ) -> RegularMemoryRecord:
        existing_ids = await self._list_ids()
        now = datetime.now(UTC)
        record = RegularMemoryRecord(
            id=next_memory_id(existing_ids),
            title=title,
            summary=summary,
            content=content,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        await self.write(record)
        return record

    async def _list_ids(self) -> set[str]:
        ids: set[str] = set()
        for directory in (self.active_dir, self.archive_dir):
            for path in directory.glob("M*.md"):
                record = await self.load(path.stem)
                if record is not None:
                    ids.add(record.id)
        return ids

    async def _get_memory_path(self, memory_id: str) -> Path | None:
        for d in (self.active_dir, self.archive_dir):
            path = d / f"{memory_id}.md"
            if await asyncio.to_thread(path.is_file) and not await asyncio.to_thread(
                path.is_symlink
            ):
                return path

        return None
