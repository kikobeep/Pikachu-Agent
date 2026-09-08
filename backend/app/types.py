"""各模块共用的数据类型。"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    role: MessageRole
    content: str | None = None
    reasoning: str | None = None
    tool_call_name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ModelUsage(BaseModel):
    """一次或多次模型调用的用量。

    ``input_tokens`` 表示 Provider 处理的全部输入（包含缓存命中）；缓存细分
    无法从响应确认时保持 ``None``，不能把“未知”伪装成 0。
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
    response: Message
    finish_reason: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw: dict[str, Any] = Field(default_factory=dict)
