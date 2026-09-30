"""工具的数据定义与执行接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.model.config import AgentMode, ToolCall


class ToolPermission(StrEnum):
    ALLOWED = "allowed"
    HUMAN_APPROVAL = "human_approval"
    FORBIDDEN = "forbidden"

    def model_visible(self) -> bool:
        """是否应该被暴露给模型"""
        return self is not ToolPermission.FORBIDDEN


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_output: bool = False
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default={"type": "object", "properties": {}}
    )
    strict: bool | None = None
    closing_allowed: bool = False
    permission: ToolPermission = ToolPermission.ALLOWED


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """单次工具调用时的运行上下文。"""

    tool_call: ToolCall
    run_id: str | None = None
    conversation_id: str | None = None
    user_input: str | None = None
    step: int | None = None
    tool_definition: ToolDefinition | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: AgentMode | None = None


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

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        """执行需要运行上下文的工具；普通工具默认复用 execute。"""

        return await self.execute(arguments)


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = Field(ge=0)

