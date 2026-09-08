from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import Message, ModelResponse
from ..tools.base import ToolDefinition
from enum import StrEnum

class ApiStyle(StrEnum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class ModelProvider(StrEnum):
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"

class ModelRequest(BaseModel):
    """可转换为任意已配置模型提供商格式的请求。"""

    model_config = ConfigDict(extra="forbid")

    messages: tuple[Message, ...]
    model: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_messages(self) -> ModelRequest:
        if not self.messages:
            raise ValueError("messages cannot be empty")
        return self

