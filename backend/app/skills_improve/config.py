"""Skill 自进化的运行配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_CONFIG_FILE = Path(__file__).resolve().parents[2] / ".config"


class SkillImprovingSettings(BaseSettings):
    """独立于主 Agent 的 Skill 改进模型与运行配置。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_CONFIG_FILE,
        env_file_encoding="utf-8",
        env_prefix="SKILL_IMPROVING_",
        extra="ignore",
    )

    enabled: bool = False
    improve_method: str = Field(default="multi_teacher", pattern="^(cluster|multi_teacher)$")
    batch_size: int = Field(default=20, ge=1)
    max_runs_per_scan: int = Field(default=100, ge=1)
    provider: str | None = None
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4_000, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_attempts: int = Field(default=2, ge=1, le=3)
    max_conversation_messages: int = Field(default=12, ge=1)
    max_feedback_chars: int = Field(default=2_000, ge=1)
    # 产出 candidate 后是否自动 accept（跳过人工 pending 审核），直接落盘正式 Skill。
    auto_accept: bool = False
    # Candidate 持久化目录（SQLite db 所在目录）。
    data_dir: str | Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / ".skills" / "improving"
        )
    )


__all__ = ["SkillImprovingSettings"]
