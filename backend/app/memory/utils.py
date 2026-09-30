"""记忆模块共用的文本、时间和文件操作。"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
import yaml

from enum import StrEnum


class MemoryStatus(StrEnum):
    """普通长期记忆的生命周期状态。"""

    ACTIVE = "active"
    ARCHIVED = "archived"


if TYPE_CHECKING:
    from app.memory.regular import RegularMemoryRecord
    from app.model.config import ModelUsage

_FRONT_MATTER_RE = re.compile(
    r"\A\ufeff?(?:[ \t]*\r?\n)*[ \t]*---[ \t]*\r?\n"
    r"(?P<metadata>.*?)^---[ \t]*(?:\r?\n|$)",
    re.DOTALL | re.MULTILINE,
)


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return " ".join(value.split())


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("text must be a string or None")
    return value.strip() or None


def normalize_memory_id(memory_id: str) -> str:
    if not isinstance(memory_id, str):
        raise TypeError("memory id must be a string")
    normalized = memory_id.strip().upper()
    if not re.fullmatch(r"M[0-9]{3,}", normalized):
        raise ValueError("memory id must match M followed by at least three digits")
    return normalized


def normalize_core_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("core memory key must be a string")
    key = value.strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("core memory key must be non-empty and contain no whitespace")
    return key


def split_front_matter(text: str) -> tuple[str | None, str]:
    matched = _FRONT_MATTER_RE.match(text)
    if matched is None:
        return None, text
    return matched.group("metadata"), text[matched.end() :].strip()


def normalize_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory datetimes must include timezone information")
    return value.astimezone(UTC)


def parse_time(value: str) -> datetime:
    return normalize_time(datetime.fromisoformat(value))


def iso_time(value: datetime) -> str:
    return normalize_time(value).isoformat()


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def estimate_tokens(content: str) -> int:
    try:
        from app.context.tokens import default_token_estimator

        return default_token_estimator().estimate_text(content)
    except (ImportError, OSError):
        return (len(content) + 1) // 2


def strip_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "\n".join(stripped.splitlines()[1:-1]).strip()
    return stripped


def add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    from app.model.config import ModelUsage

    values = {}
    for name in ModelUsage.model_fields:
        a, b = getattr(left, name), getattr(right, name)
        values[name] = None if a is None or b is None else a + b
    return ModelUsage(**values)


def next_memory_id(existing_ids: set[str]) -> str:
    """分配下一个形如 M001 的 Memory ID（在现有 ID 之后递增）。"""

    highest = 0
    for memory_id in existing_ids:
        suffix = memory_id.removeprefix("M")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"M{highest + 1:03d}"


def parse_memory_markdown(text: str) -> RegularMemoryRecord:
    front, body = split_front_matter(text)
    if front is None:
        raise ValueError("memory file is missing YAML front matter")
    data = yaml.safe_load(front)
    if not isinstance(data, dict):
        raise ValueError("memory front matter must be a mapping")
    lines = body.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "## Memory"
        )
    except StopIteration:
        raise ValueError("memory file is missing the Memory section") from None
    data["content"] = "\n".join(lines[start + 1 :]).strip()
    for field in ("created_at", "updated_at", "last_accessed_at"):
        data[field] = parse_time(str(data[field]))
    from app.memory.regular import RegularMemoryRecord

    return RegularMemoryRecord.model_validate(data)


_MAX_MEMORY_FILE_BYTES = 512_000
def convert_to_record(path: Path) -> RegularMemoryRecord:
    if path.stat().st_size > _MAX_MEMORY_FILE_BYTES:
        raise ValueError(f"memory file too large: {path.name}")
    text = path.read_text(encoding="utf-8")
    record = parse_memory_markdown(text)
    if record.id != path.stem:
        raise ValueError(
            f"memory id does not match filename: {record.id} != {path.stem}"
        )
    expected_status = (
        MemoryStatus.ARCHIVED
        if path.parent.name == "archive"
        else MemoryStatus.ACTIVE
    )
    if record.status is not expected_status:
        raise ValueError(
            f"memory status does not match directory: {record.status.value}"
        )
    return record