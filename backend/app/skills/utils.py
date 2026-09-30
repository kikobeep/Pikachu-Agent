from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict
from .config import SKILL_DESCRIPTION_MAX_LENGTH

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_NAME_MAX_LENGTH = 64
SKILL_FILE_NAME = "SKILL.md"

def validate_skill_name(name: str) -> str:
    """校验 Skill name；非法时抛出 ValueError。"""

    normalized = name.strip()
    if not normalized:
        raise ValueError("skill name cannot be empty")
    if len(normalized) > SKILL_NAME_MAX_LENGTH:
        raise ValueError(
            f"skill name exceeds {SKILL_NAME_MAX_LENGTH} chars: {name!r}"
        )
    if not _SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "skill name must be lowercase letters/digits separated by single "
            f"hyphens: {name!r}"
        )
    return normalized


def check_skill_dir(root: str | Path, name: str) -> Path | None:
    """返回位于 root 内、非符号链接的 Skill 目录；非法则返回 None。"""

    try:
        validate_skill_name(name)
    except ValueError:
        return None
    root_path = Path(root).expanduser().resolve()
    candidate = root_path / name
    if candidate.is_symlink():
        return None
    resolved = candidate.resolve()
    if not is_relative(root_path, resolved):
        return None
    if not resolved.is_dir():
        return None
    return resolved

def safe_skill_file(skill_dir: Path) -> Path | None:
    """返回 Skill 目录内的普通（非符号链接）SKILL.md；非法返回 None。"""

    skill_file = skill_dir / SKILL_FILE_NAME
    if skill_file.is_symlink():
        return None
    if not skill_file.is_file():
        return None
    resolved = skill_file.resolve()
    if not is_relative(skill_dir.resolve(), resolved):
        return None
    return skill_file


def is_relative(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False



_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "allowed_tools",
    }
)


class SkillParseError(ValueError):
    """SKILL.md 解析或结构校验失败。"""


class ParsedSkill(BaseModel):
    """解析出的结构化 Skill 元数据与正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, object] | None = None
    allowed_tools: tuple[str, ...] = ()
    body: str


def parse_skill_document(text: str, *, expected_name: str) -> ParsedSkill:
    """解析 SKILL.md；任何非法输入抛出 SkillParseError。"""

    front, body = _split_front_matter(text)
    if front is None:
        raise SkillParseError("missing YAML front matter")
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillParseError("front matter must be a mapping")

    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL_FIELDS)
    if unknown:
        raise SkillParseError(
            f"unknown front matter field(s): {', '.join(unknown)}"
        )

    name = data.get("name")
    if not isinstance(name, str):
        raise SkillParseError("missing or non-string 'name'")
    try:
        normalized_name = validate_skill_name(name)
    except ValueError as exc:
        raise SkillParseError(str(exc)) from exc
    if normalized_name != expected_name:
        raise SkillParseError(
            f"front matter name '{normalized_name}' does not match "
            f"directory name '{expected_name}'"
        )

    description = data.get("description")
    if not isinstance(description, str):
        raise SkillParseError("missing or non-string 'description'")
    description = description.strip()
    if not description:
        raise SkillParseError("empty 'description'")
    if len(description) > SKILL_DESCRIPTION_MAX_LENGTH:
        raise SkillParseError(
            f"'description' exceeds {SKILL_DESCRIPTION_MAX_LENGTH} chars"
        )

    license_value = _optional_str(data, "license", "license")
    compatibility = _optional_str(data, "compatibility", "compatibility")

    metadata_value = data.get("metadata")
    if metadata_value is not None and not isinstance(metadata_value, dict):
        raise SkillParseError("'metadata' must be a mapping")

    allowed_tools = _allowed_tools(data)
    if not body.strip():
        raise SkillParseError("empty skill body")

    return ParsedSkill(
        name=normalized_name,
        description=description,
        license=license_value,
        compatibility=compatibility,
        metadata=metadata_value,
        allowed_tools=allowed_tools,
        body=body.strip(),
    )


def _optional_str(
    data: dict[str, Any],
    key: str,
    label: str,
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SkillParseError(f"'{label}' must be a string")
    stripped = value.strip()
    if not stripped:
        raise SkillParseError(f"'{label}' cannot be empty")
    return stripped


def _allowed_tools(data: dict[str, Any]) -> tuple[str, ...]:
    hyphen_key = data.get("allowed-tools")
    underscore_key = data.get("allowed_tools")
    if hyphen_key is not None and underscore_key is not None:
        raise SkillParseError(
            "'allowed-tools' and 'allowed_tools' cannot both be present"
        )
    raw = hyphen_key if hyphen_key is not None else underscore_key
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        raise SkillParseError("'allowed-tools' must be a list of non-empty strings")
    return tuple(str(item).strip() for item in raw)


def _split_front_matter(text: str) -> tuple[str | None, str]:
    """把 ``---`` 包裹的 Front Matter 与正文分开。"""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :]).strip()
    return None, text


__all__ = ["ParsedSkill", "SkillParseError", "parse_skill_document"]
