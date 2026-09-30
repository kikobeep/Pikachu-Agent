from pathlib import Path
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"

MAX_SKILL_FILE_BYTES = 512000
SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024
DEFAULT_USER_SKILLS_DIR = Path.home() / ".skills" / "skills"
DEFAULT_PROJECT_SKILLS_DIR = (Path(__file__).resolve().parents[2] / ".skills" / "skills")


class SkillScope(StrEnum):
    """Skill 的来源层级。"""

    USER = "user"
    PROJECT = "project"


class SkillMetadata(BaseModel):
    """Skill 的轻量目录项（Discovery 阶段建立，不包含正文）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    scope: SkillScope
    location: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, object] | None = None
    # TODO(skill-allowed-tools): 尚未参与工具权限。未来只允许收窄当前 Run 的
    # 工具集合（不能把 approval 提升成 allowed，也不能解禁 forbidden）；
    # 在 Permission/ToolExecutor 支持该不变量前，保持"只解析、不生效"。
    allowed_tools: tuple[str, ...] = ()

    def render_catalog_entry(self) -> str:
        """渲染为注入模型上下文的精简目录项（仅 name + description）。"""

        return f"[{self.name}] {self.description}"


class SkillResources(BaseModel):
    """Skill 目录内可安全访问的资源清单（仅相对路径，不加载正文）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scripts: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "references": self.references,
            "scripts": self.scripts,
            "assets": self.assets,
        }

    def is_empty(self) -> bool:
        return not (self.scripts or self.references or self.assets)


class Skill(BaseModel):
    """激活后的完整 Skill（正文 + 资源清单 + 来源根目录）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: SkillMetadata
    content: str
    root: Path
    resources: SkillResources = Field(default_factory=SkillResources)

    def render_instructions(self) -> str:
        """渲染为注入模型上下文的 Active Skill 指令块。"""

        header = f"# Skill: {self.metadata.name}"
        body = [
            header,
            "",
            self.content.strip(),
        ]
        if not self.resources.is_empty():
            body.extend(
                (
                    "",
                    "## Resources",
                    "",
                    "可用的资源（需要时用 skill_resource_read 读取，不自动加载）：",
                )
            )
            for kind, items in self.resources.as_dict().items():
                if items:
                    body.append(f"- {kind}: " + ", ".join(items))
        return "\n".join(body)

class SkillSettings(BaseSettings):
    """Skill 上下文预算等运行配置。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Active Skill 指令的总上下文预算（token）。
    skill_context_max_tokens: int = Field(default=4_096, gt=0)
    # 同一 Run 最多同时激活的 Skill 数量。
    skill_max_active: int = Field(default=4, gt=0)
    # Skill Catalog（name + description）每 Step 注入的独立 Token 预算。
    skill_catalog_max_tokens: int = Field(default=2_048, gt=0)