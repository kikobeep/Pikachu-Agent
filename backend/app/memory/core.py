from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from .utils import (
    estimate_tokens,
    normalize_core_key,
    normalize_text,
    normalize_time,
    split_front_matter,
    write_atomic,
)

_CORE_HEADING = "# Core Memory"
_CORE_FORMAT = "core-memory-v1"
# Core 条目的默认长度限制。
_MAX_CORE_VALUE_CHARS = 500
_MAX_CORE_REASON_CHARS = 500
_MAX_SOURCE_STATEMENT_CHARS = 1000


class CoreMemoryRecord(BaseModel):
    """Harness 按稳定 key 管理的一条 Core Memory。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    value: str
    reason: str
    source_statement: str
    updated_at: datetime

    @field_validator("key", mode="before")
    @classmethod
    def normalize_key(cls, value: object) -> str:
        return normalize_core_key(value)

    @field_validator("value", "reason", "source_statement", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("core memory value and reason must be strings")
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("core memory value and reason cannot be empty")
        return normalized

    @field_validator("value")
    @classmethod
    def validate_value_length(cls, value: str) -> str:
        if len(value) > _MAX_CORE_VALUE_CHARS:
            raise ValueError(
                f"core memory value exceeds {_MAX_CORE_VALUE_CHARS} characters"
            )
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason_length(cls, value: str) -> str:
        if len(value) > _MAX_CORE_REASON_CHARS:
            raise ValueError(
                f"core memory reason exceeds {_MAX_CORE_REASON_CHARS} characters"
            )
        return value

    @field_validator("source_statement")
    @classmethod
    def validate_source_statement_length(cls, value: str) -> str:
        if len(value) > _MAX_SOURCE_STATEMENT_CHARS:
            raise ValueError(
                "core memory source statement exceeds "
                f"{_MAX_SOURCE_STATEMENT_CHARS} characters"
            )
        return value

    @field_validator("updated_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return normalize_time(value)


class CoreMemory:
    def __init__(self, memory_dir: str | Path, max_tokens: int = 2000) -> None:
        self.path = Path(memory_dir) / "CORE.md"
        self.max_tokens = max_tokens

    async def initialize(self) -> None:
        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)

    async def load(self) -> str | None:
        await self.initialize()
        if not await asyncio.to_thread(self.path.is_file):
            return
        content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")

        tokens = estimate_tokens(content)
        if tokens > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {tokens} > {self.max_tokens}"
            )
        return content

    async def update(self, content: str) -> None:
        """更新 CORE.md，覆盖式写入非增量更新"""
        normalized = content.strip()
        if not normalized:
            raise ValueError("core memory content cannot be empty")
        estimated = estimate_tokens(normalized)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )
        await asyncio.to_thread(write_atomic, self.path, normalized + "\n")

    async def insert(
        self,
        *,
        key: str,
        value: str,
        reason: str,
        source_statement: str,
    ) -> tuple[CoreMemoryRecord, bool]:
        """更新 CORE.md，条目式写入，增量更新"""
        now = datetime.now(UTC)
        new_item = CoreMemoryRecord(
            key=key,
            value=value,
            reason=reason,
            source_statement=source_statement,
            updated_at=now,
        )
        raw_content = ""
        if await asyncio.to_thread(self.path.is_file):
            if await asyncio.to_thread(self.path.is_symlink):
                raise ValueError("CORE.md cannot be a symbolic link")
            raw_content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")

        # 得到条目和正文
        items, content = _parse_document(raw_content)
        created = new_item.key not in items
        # 把最新的记忆加入到现有条目中
        items[new_item.key] = new_item
        rendered, body = render_document(items, content)

        estimated = estimate_tokens(rendered)
        if estimated > self.max_tokens:
            raise ValueError(
                f"core memory exceeds token limit: {estimated} > {self.max_tokens}"
            )

        await asyncio.to_thread(write_atomic, self.path, rendered)
        return new_item, created

    async def remove(self, key: str) -> CoreMemoryRecord:
        """按 key 移除一条 Core Memory 条目并重新渲染。"""

        normalized_key = normalize_core_key(key)
        raw_content = ""
        if await asyncio.to_thread(self.path.is_file):
            if await asyncio.to_thread(self.path.is_symlink):
                raise ValueError("CORE.md cannot be a symbolic link")
            raw_content = await asyncio.to_thread(
                self.path.read_text, encoding="utf-8"
            )

        items, content = _parse_document(raw_content)
        if normalized_key not in items:
            raise KeyError(f"core memory key not found: {key}")
        removed = items.pop(normalized_key)
        rendered, _ = render_document(items, content)
        await asyncio.to_thread(write_atomic, self.path, rendered)
        return removed


def _parse_document(text: str) -> tuple[dict[str, CoreMemoryRecord], str]:
    """解析结构化 Core；旧的纯 Markdown 文件作为可保留正文处理。"""
    front, body = split_front_matter(text)
    if front is None:
        return {}, text.strip()
    metadata = yaml.safe_load(front)
    if not isinstance(metadata, dict):
        raise ValueError("CORE.md metadata must be a mapping")
    raw_items = metadata.get("entries", [])
    if not isinstance(raw_items, list):
        raise ValueError("CORE.md entries metadata must be a list")
    items = {}
    for raw in raw_items:
        entry = CoreMemoryRecord.model_validate(raw)
        items[entry.key] = entry
    # 本模块生成的正文会随条目重新渲染；另存旧的自由 Markdown 正文。
    legacy_body = metadata.get(
        "legacy_body", "" if metadata.get("format") == _CORE_FORMAT else body
    )
    return items, legacy_body


def render_document(
    items: dict[str, CoreMemoryRecord], legacy_body: str = ""
) -> tuple[str, str]:
    sorted_entries = sorted(
        items.values(),
        key=lambda entry: entry.key,
    )
    entry_list = []

    for entry in sorted_entries:
        entry_data = {
            "key": entry.key,
            "value": entry.value,
            "reason": entry.reason,
            "source_statement": entry.source_statement,
            "updated_at": entry.updated_at.isoformat(timespec="seconds"),
        }
        entry_list.append(entry_data)

    metadata = {
        "format": _CORE_FORMAT,
        "legacy_body": legacy_body,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "entries": entry_list,
    }

    front = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
    )
    body: list[str] = [_CORE_HEADING, ""]
    for entry in sorted_entries:
        body.extend((f"### {entry.key}", "", entry.value, ""))
    if legacy_body:
        body.extend((legacy_body, ""))
    body = "\n".join(body).rstrip() + "\n"
    return f"---\n{front}---\n{body}", body
