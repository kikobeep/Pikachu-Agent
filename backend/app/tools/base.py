"""工具的数据定义与执行接口。"""
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


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: bool 
    output: Any = None
    error: str | None = None
    duration_ms: float = Field(ge=0)
