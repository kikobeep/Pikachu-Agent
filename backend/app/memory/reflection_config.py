from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from app.model.config import ModelUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import normalize_memory_id, normalize_optional_text, normalize_text

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"


class ReflectionAction(StrEnum):
    NONE = "none"
    CREATE = "create"
    UPDATE = "update"


class MemoryReflectionConfig(BaseSettings):
    """独立于主 Agent 的 Reflection 模型配置。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_REFLECTION_",
        extra="ignore",
    )

    enabled: bool = True
    provider: str | None = None
    model: str | None = None
    max_output_tokens: int = Field(default=2_000, gt=0)
    max_attempts: int = Field(default=2, ge=1, le=2)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_tool_context_chars: int = Field(default=8_000, ge=0, le=20_000)
    capture_raw_io: bool = False
    gate_enabled: bool = True

    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        return normalize_optional_text(value)


class MemoryReflectionInput(BaseModel):
    """Reflector 可见的本轮有界上下文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    conversation_id: str | None = None
    user_input: str
    final_answer: str
    tool_context: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    core_memory: str = ""
    memory_index: str = ""
    task_context: str = ""

    @field_validator("memory_ids")
    @classmethod
    def normalize_recalled_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            memory_id = normalize_memory_id(value)
            if memory_id not in normalized:
                normalized.append(memory_id)
        return tuple(normalized)


class ReflectionDecision(BaseModel):
    """严格单动作输出；自由文本不能直接写入 Memory Store。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ReflectionAction
    memory_id: str | None = None
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    reason: str

    @field_validator(
        "memory_id",
        "title",
        "summary",
        "content",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        return normalize_optional_text(value)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("reflection reason must be a string")
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("reflection reason cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_action_fields(self) -> ReflectionDecision:
        if self.action is ReflectionAction.NONE:
            if any(
                value is not None
                for value in (
                    self.memory_id,
                    self.title,
                    self.summary,
                    self.content,
                )
            ):
                raise ValueError("none decision cannot contain mutation fields")
            return self
        if self.action is ReflectionAction.CREATE:
            if self.memory_id is not None:
                raise ValueError("create decision cannot contain memory_id")
            if not self.title or not self.summary or not self.content:
                raise ValueError("create decision requires title, summary, and content")
            return self
        if not self.memory_id:
            raise ValueError("update decision requires a valid memory_id")
        if not self.title or not self.summary or not self.content:
            raise ValueError(
                "update decision requires title, summary, and complete content"
            )
        return self.model_copy(
            update={"memory_id": normalize_memory_id(self.memory_id)}
        )


class MemoryReflectionReponse(BaseModel):
    """Reflection 模型调用结果；是否真正写入由 Runtime 协调。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ReflectionDecision | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    attempts: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    error: str | None = None
    input_json: str | None = None
    raw_output: str | None = None
