"""仅允许读取 Vesta 工作区的 UTF-8 文本工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.models.types import ToolDefinition

from ..base import BaseTool
from ._workspace import resolve_workspace_path, workspace_root_path


class ReadFileTool(BaseTool):
    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = workspace_root_path(workspace_root)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="读取本地工作区中的 UTF-8 文本文件。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作区的文件路径。",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            strict=True,
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        relative_path = arguments.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("'path' must be a non-empty string")
        target = resolve_workspace_path(self._workspace_root, relative_path)
        return await asyncio.to_thread(_read_utf8_file, target)


def _read_utf8_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path.name}")
    return path.read_text(encoding="utf-8")
