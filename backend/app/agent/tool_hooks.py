"""将工具执行和审批阶段转换为事件通知。"""

from __future__ import annotations

from typing import Any, Protocol

from app.tools import ToolResult
from app.tools.approval import ApprovalRequest
from app.tools.config import ToolExecutionContext


class AgentEventEmitter(Protocol):
    """工具事件通知所需的最小接口；具体输出由调用方实现。"""

    async def emit(self, event_type: str, **payload: Any) -> None:
        ...


class AgentEventHook:
    """报告工具执行过程，相比于agentevent，hook只工具执行和审批相关的事件"""

    def __init__(self, emitter: AgentEventEmitter) -> None:
        self._emitter = emitter

    async def before_execute(self, context: ToolExecutionContext) -> None:
        await self._emitter.emit(
            "tool_started",
            step=context.step,
            tool_call=context.tool_call,
        )

    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        await self._emitter.emit(
            "tool_approval_required",
            step=context.step,
            tool_call=context.tool_call,
        )

    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: bool,
    ) -> None:
        await self._emitter.emit(
            "tool_approval_completed",
            step=context.step,
            tool_call=context.tool_call,
            approval_decision=decision,
        )

    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        await self._emitter.emit(
            "tool_completed",
            step=context.step,
            tool_call=context.tool_call,
            tool_result=result,
        )


__all__ = ["AgentEventEmitter", "AgentEventHook"]
