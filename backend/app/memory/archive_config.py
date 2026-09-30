from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from app.model.config import ModelUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .regular import RegularMemoryRecord
from .utils import normalize_memory_id

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"


class ArchiveAction(StrEnum):
    """容量维护允许执行的单动作。"""

    ARCHIVE = "archive"
    DEFER = "defer"


class MemoryArchiveDecision(BaseModel):
    """维护模型的严格 archive/defer 单动作输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ArchiveAction
    memory_id: str | None = None
    reason: str

    @field_validator("memory_id", mode="before")
    @classmethod
    def normalize_optional_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("archive memory_id must be a string")
        return normalize_memory_id(value)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("archive reason must be a string")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("archive reason cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_action_fields(self) -> MemoryArchiveDecision:
        if self.action is ArchiveAction.ARCHIVE and self.memory_id is None:
            raise ValueError("archive decision requires memory_id")
        if self.action is ArchiveAction.DEFER and self.memory_id is not None:
            raise ValueError("defer decision cannot contain memory_id")
        return self


class MemoryArchiveConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_ARCHIVE_",
        extra="ignore",
    )
    enabled: bool = True
    provider: str | None = None
    model: str | None = None
    max_output_tokens: int = Field(default=800, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    candidate_limit: int = Field(default=5, ge=1, le=10)
    max_actions: int = Field(default=5, ge=1, le=20)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("archive provider and model must be strings")
        return value.strip() or None


class MemoryArchiveInput(BaseModel):
    """一次容量决策所需的有界候选集合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_count: int = Field(ge=0)
    max_active: int = Field(gt=0)
    required_slots: int = Field(default=0, ge=0)
    candidates: tuple[RegularMemoryRecord, ...]


class MemoryArchiveResponse(BaseModel):
    """维护模型调用结果；文件 mutation 仍由 Harness 执行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: MemoryArchiveDecision | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    error: str | None = None
