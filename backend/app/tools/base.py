"""工具的数据定义与执行接口。"""
from dataclasses import dataclass, field
from app.types import AgentMode, ToolCall
from enum import StrEnum
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolPermission(StrEnum):
    ALLOWED = "allowed"
    HUMAN_APPROVAL = "human_approval"
    FORBIDDEN = "forbidden"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default={"type": "object", "properties": {}}
    )
    closing_allowed: bool = False
    permission: ToolPermission = ToolPermission.ALLOWED


class BaseTool(ABC):
    """具体工具需要提供定义并实现异步执行。"""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回工具说明及参数结构。"""
        ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """执行工具并返回结果。"""
        ...


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: bool 
    output: Any = None
    error: str | None = None
    duration_ms: float = Field(ge=0)

@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """一次工具调用在执行链中的共享上下文。"""

    tool_call: ToolCall
    run_id: str | None = None
    conversation_id: str | None = None
    user_input: str | None = None
    step: int | None = None
    tool_definition: ToolDefinition | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: AgentMode | None = None
