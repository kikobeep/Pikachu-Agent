
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.tools.config import ToolDefinition

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"

class ApiStyle(StrEnum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class ModelProvider(StrEnum):
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"


class ProviderConfig(BaseModel):
    """用于创建单个适配器的完整配置。"""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_key: SecretStr
    api_style: ApiStyle
    base_url: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    default_max_output_tokens: int = Field(default=4096, gt=0)

    def api_key_value(self) -> str:
        return self.api_key.get_secret_value()

class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
    model_default_provider: ModelProvider = ModelProvider.OPENAI
    model_timeout_seconds: float = Field(default=120.0, gt=0)
    model_max_retries: int = Field(default=2, ge=0)
    model_default_max_output_tokens: int = Field(default=4096, gt=0)


    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str | None = None
    openai_api_style: ApiStyle = ApiStyle.RESPONSES

    qwen_api_key: SecretStr | None = None
    qwen_model: str = "qwen3.7-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_style: ApiStyle = ApiStyle.CHAT_COMPLETIONS

    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_style: ApiStyle = ApiStyle.CHAT_COMPLETIONS
    
    
    def load_provider_config(
        self,
        model: ModelProvider | str,
    ) -> ProviderConfig:
        provider = ModelProvider(model)

        configs = {
            ModelProvider.OPENAI: {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                "api_style": self.openai_api_style,
            },
            ModelProvider.QWEN: {
                "api_key": self.qwen_api_key,
                "model": self.qwen_model,
                "base_url": self.qwen_base_url,
                "api_style": self.qwen_api_style,
            },
            ModelProvider.DEEPSEEK: {
                "api_key": self.deepseek_api_key,
                "model": self.deepseek_model,
                "base_url": self.deepseek_base_url,
                "api_style": self.deepseek_api_style,
            },
        }
        
        return self._build_config(
                provider=provider,
                **configs[provider]
            )
    
    def _build_config(
        self,
        provider: ModelProvider,
        api_key: SecretStr | None,
        model: str,
        base_url: str | None,
        api_style:ApiStyle
    ) -> ProviderConfig:
        if api_key is None or not api_key.get_secret_value():
            raise ValueError(f"Provider '{provider.value}' is not configured.")
        return ProviderConfig(
            provider=provider.value,
            model=model,
            api_key=api_key,
            api_style=api_style,
            base_url=base_url or None,
            timeout_seconds=self.model_timeout_seconds,
            max_retries=self.model_max_retries,
            default_max_output_tokens=self.model_default_max_output_tokens,
        )

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

class ModelUsage(BaseModel):
    """一次或多次模型调用的用量。
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int | None = Field(default=None, ge=0)
    uncached_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    model_calls: int = Field(default=0, ge=0)


class ModelResponse(BaseModel):
    """所有模型适配器统一返回的结果。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: str
    model: str
    message: Message
    finish_reason: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


class AgentMode(StrEnum):
    DEFAULT = "default"
    PLAN = "plan"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class Message(BaseModel):
    name: str | None = None
    role: MessageRole
    content: str | None = None
    reasoning: str | None = None
    tool_call_name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


def rebuild_models() -> None:
    """重建包含跨模块前向引用的模型。

    ModelRequest 依赖 Message / ToolCall；ToolCall 在本模块，但 ModelRequest.tools
    里出现的 ToolDefinition 来自 app.tools.config，不能在 config.py 内 rebuild。
    调用前需要先 import app.tools.config 以将 ToolDefinition 注入模块 globals。
    """

    ModelRequest.model_rebuild()


# 在 module 末尾惰性 import：此时 app.tools.config 已全部载入，ToolDefinition
# 已绑定到全局命名空间，可以安全地触发跨模块前向引用重建。
try:
    from app.tools.config import ToolDefinition  # noqa: F401
    ModelRequest.model_rebuild()
except ImportError:
    # 第一次被 tools.config 反向 import 时 app.tools.config 尚未完成，跳过。
    pass
