from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils import (
    MemoryStatus,
    iso_time,
    normalize_text,
    normalize_time,
    parse_time,
    split_front_matter,
    write_atomic,
)

# 记忆字段的默认长度限制。
_MAX_TITLE_CHARS = 120
_MAX_SUMMARY_CHARS = 300
_MAX_CONTENT_CHARS = 12000
_MAX_REASON_CHARS = 500

_INDEX_HEADER = (
    "# Long-term Memory Index\n\n"
    "The following long-term memories are available.\n"
    "These cues are discovery metadata, not authoritative memory content.\n"
    "When a memory may materially help the current task, use memory_read "
    "before relying on it in an answer, decision, or action.\n"
)

class RegularMemoryRecord(BaseModel):
    """一条普通长期记忆及其运行时元数据。

    运行时字段（``created_at``/``updated_at``/``last_accessed_at``/
    ``access_count``/``status``）由 Memory Store 维护，模型不能自行填写。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^M[0-9]{3,}$")
    title: str
    summary: str
    content: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime
    access_count: int = Field(default=0, ge=0)
    revision: int = Field(default=1, ge=1)
    status: MemoryStatus = MemoryStatus.ACTIVE
    last_update_reason: str | None = None
    archive_reason: str | None = None

    @field_validator("title", "summary", mode="before")
    @classmethod
    def normalize_cue_text(cls, value: object) -> str:
        """标题和 Recall Cue 必须紧凑，避免 INDEX 被长文本撑大。"""

        if not isinstance(value, str):
            raise TypeError("memory title and summary must be strings")
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("memory title and summary cannot be empty")
        return normalized

    @field_validator("title")
    @classmethod
    def validate_title_length(cls, value: str) -> str:
        if len(value) > _MAX_TITLE_CHARS:
            raise ValueError(f"memory title exceeds {_MAX_TITLE_CHARS} characters")
        return value

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, value: str) -> str:
        if len(value) > _MAX_SUMMARY_CHARS:
            raise ValueError(f"memory summary exceeds {_MAX_SUMMARY_CHARS} characters")
        return value

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("memory content must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory content cannot be empty")
        if len(normalized) > _MAX_CONTENT_CHARS:
            raise ValueError(f"memory content exceeds {_MAX_CONTENT_CHARS} characters")
        return normalized

    @field_validator("last_update_reason", "archive_reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("memory change reason must be a string")
        normalized = normalize_text(value)
        if not normalized:
            return None
        if len(normalized) > _MAX_REASON_CHARS:
            raise ValueError(
                f"memory change reason exceeds {_MAX_REASON_CHARS} characters"
            )
        return normalized

    @field_validator("created_at", "updated_at", "last_accessed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return normalize_time(value)

    def front_matter(self) -> dict[str, object]:
        """序列化为 Front Matter 字典（运行时元数据）。"""

        metadata: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "created_at": iso_time(self.created_at),
            "updated_at": iso_time(self.updated_at),
            "last_accessed_at": iso_time(self.last_accessed_at),
            "access_count": self.access_count,
            "revision": self.revision,
            "status": self.status.value,
        }
        if self.last_update_reason is not None:
            metadata["last_update_reason"] = self.last_update_reason
        if self.archive_reason is not None:
            metadata["archive_reason"] = self.archive_reason
        return metadata

    def render_markdown(self) -> str:
        """渲染为带 Front Matter 的 Markdown 文件内容。"""

        front = yaml.safe_dump(
            self.front_matter(),
            allow_unicode=True,
            sort_keys=False,
        )
        body = "\n".join(
            (
                f"# {self.title}",
                "",
                "## Summary",
                "",
                self.summary,
                "",
                "## Memory",
                "",
                self.content,
                "",
            )
        )
        return f"---\n{front}---\n{body}"

    def render_full(self) -> str:
        """渲染为模型 ``memory_read`` 可见的完整正文（不含 Front Matter）。"""

        return (
            f"# {self.title}\n\n"
            f"## Summary\n\n{self.summary}\n\n"
            f"## Memory\n\n{self.content}"
        )


class RegularMemoryIndex:
    def __init__(self, memory_dir):
        self.path = Path(memory_dir) / "Regular.md"

    async def load(self) -> str | None:
        if not await asyncio.to_thread(self.path.is_file):
            return None
        return await asyncio.to_thread(self.path.read_text, encoding="utf-8")

    async def rebuild(self, memories: Sequence[RegularMemoryRecord]) -> None:

        content = self.render(memories)
        await asyncio.to_thread(write_atomic, self.path, content)

    def render(self, memories: Sequence[RegularMemoryRecord]) -> str:
        if not memories:
            return _INDEX_HEADER + "\n(No long-term memories yet.)\n"
        lines = [_INDEX_HEADER]
        for record in memories:
            summary = " ".join(record.summary.split()).strip()
            cue = summary or record.title
            lines.append(f"[{record.id}] {record.title}")
            lines.append(f"Cue: {cue}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"



