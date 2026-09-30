from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.config import BaseTool, ToolDefinition
from app.tools.register import ToolRegistry

from .store import SkillStore

SKILL_READ_TOOL_NAME = "skill_read"
SKILL_RESOURCE_READ_TOOL_NAME = "skill_resource_read"

_MAX_RESOURCE_BYTES = 64_000


class SkillReadTool(BaseTool):
    """请求激活一个 Skill（轻量：不返回完整正文）。

    完整指令只会在 Runtime 成功激活后，通过 ``ctive_skill`` 系统
    消息在每个 Agent Step 注入；此处只返回 metadata 与资源清单。
    """

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="skill_read",
            record_output=False,
            description=(
                "请求激活一个 Skill。只有当 Available Skills 目录中出现当前"
                "任务匹配的 Skill 时才调用；激活后其完整指令会在本 Run 内"
                "持续注入（不再返回指令正文）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "要激活的 Skill 名称（Available Skills 目录中的 name）。"
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        skill = await self._store.load(name)
        if skill is None:
            return {"found": False, "name": name}
        return {
            "found": True,
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "scope": skill.metadata.scope.value,
            "resources": skill.resources.as_dict(),
        }


class SkillResourceReadTool(BaseTool):
    """安全读取当前 Run 已激活 Skill 目录内的资源。

    只允许读取本 Run 已激活（active_skills）的 Skill 资源；未激活的 Skill
    会被拒绝。路径安全规则与文件大小限制继续生效。
    """

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="skill_resource_read",
            record_output=False,
            description=(
                "读取当前已激活 Skill 目录内的一个资源文件（Active Skill 指令"
                "列出的 references/scripts/assets 相对路径）。只能读取本 Run "
                "已通过 skill_read 激活的 Skill；路径必须位于该 Skill 目录内，"
                "禁止 ../ 或绝对路径。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill 名称。",
                    },
                    "path": {
                        "type": "string",
                        "description": "Skill 内资源的相对路径。",
                    },
                },
                "required": ["name", "path"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """无 Run 上下文的直接调用（兼容直接 execute 场景）。"""

        return await self._read_resource(arguments)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """Run-scoped 调用：只允许读取当前 Run 已激活 Skill 的资源。"""

        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        active_names = context.metadata.get("active_skill_names", ())
        if name not in active_names:
            return {
                "found": False,
                "name": name,
                "error": (
                    "skill is not active in the current run; call skill_read first"
                ),
            }
        return await self._read_resource(arguments)

    async def _read_resource(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name")
        resource_path = arguments.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")
        if not isinstance(resource_path, str) or not resource_path.strip():
            raise ValueError("'path' must be a non-empty string")
        skill = await self._store.load(name)
        if skill is None:
            return {"found": False, "name": name}
        target = load_skill_resource(skill.root, resource_path)
        if target is None:
            return {
                "found": False,
                "name": name,
                "path": resource_path,
                "error": "resource path escapes skill root or is not a file",
            }
        try:
            if target.stat().st_size > _MAX_RESOURCE_BYTES:
                return {
                    "found": False,
                    "name": name,
                    "path": resource_path,
                    "error": f"resource exceeds {_MAX_RESOURCE_BYTES} bytes",
                }
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return {
                "found": False,
                "name": name,
                "path": resource_path,
                "error": "resource read failed",
            }
        return {
            "found": True,
            "name": name,
            "path": resource_path,
            "content": content,
        }


def register_skill_tools(
    registry: ToolRegistry,
    store: SkillStore,
) -> None:
    """把 Skill 工具注册到本地工具注册表。"""

    registry.register(SkillReadTool(store))
    registry.register(SkillResourceReadTool(store))


def load_skill_resource(skill_dir: Path, relative: str) -> Path | None:
    root = skill_dir.resolve()
    target = (root / relative).resolve()
    # 禁止 ../、绝对路径与越界 symlink：解析后必须仍在 skill 根目录内
    if not target.is_relative_to(root):
        return None
    return target if target.is_file() else None

__all__ = [
    "SKILL_READ_TOOL_NAME",
    "SKILL_RESOURCE_READ_TOOL_NAME",
    "SkillReadTool",
    "SkillResourceReadTool",
    "register_skill_tools",
]
