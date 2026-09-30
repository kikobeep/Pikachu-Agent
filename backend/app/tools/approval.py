"""人工审批请求与可替换的审批入口。"""

import asyncio
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from enum import StrEnum      
from pydantic import BaseModel, ConfigDict 

class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalScope(StrEnum):
    """审批通过后规则的作用范围。"""

    ONCE = "once"  # 仅允许这一次，不创建规则
    RUN = "run"  # 当前 Run 内允许完全相同的操作
    CONVERSATION = "conversation"  # 当前会话内记住完全相同的操作


class ApprovalResponse(BaseModel):
    """ApprovalGate 返回的用户选择。"""

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision
    scope: ApprovalScope = ApprovalScope.ONCE



@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """人工审核上下文；参数与调用方隔离，执行器会核对审批前后的一致性。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str = ""
    run_id: str | None = None
    conversation_id: str | None = None
    ui_scope: str = "sandbox"

    def summary(self, *, max_arguments: int | None = 500) -> str:
        if max_arguments is not None and max_arguments < 0:
            raise ValueError("max_arguments must be non-negative")
        serialized = json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if max_arguments is not None and len(serialized) > max_arguments:
            serialized = serialized[:max_arguments] + "…"
        return "\n".join([
            f"工具: {self.tool_name}",
            f"说明: {self.description or '(无)'}",
            f"参数: {serialized}",
        ])


class ApprovalGate(ABC):
    """决定 HUMAN_APPROVAL 工具是否可以执行。"""

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """返回用户的选择（决定 + 可选的作用范围）。"""


class DenyAllGate(ApprovalGate):
    """拒绝所有审核请求（安全默认值）。"""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(decision=ApprovalDecision.DENIED)

class AutoApproveGate(ApprovalGate):
    """无人值守场景（eval / 沙箱）自动批准审批请求。

    approve_tool_names 为空则批准所有 HUMAN_APPROVAL 工具；
    否则只批准名单内的工具，其余仍拒绝。
    """

    def __init__(self, approve_tool_names: tuple[str, ...] | None = None) -> None:
        self._approve_tool_names = (
            set(approve_tool_names) if approve_tool_names else None
        )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        if (
            self._approve_tool_names is None
            or request.tool_name in self._approve_tool_names
        ):
            return ApprovalResponse(decision=ApprovalDecision.APPROVED)
        return ApprovalResponse(decision=ApprovalDecision.DENIED)

class ConsoleApprovalGate:
    """命令行审批；完整展示参数，只接受 y/yes，输入结束视为拒绝。"""

    def __init__(
        self,
        *,
        prompt_prefix: str = "[人工审核]",
    ) -> None:
        self._prompt_prefix = prompt_prefix
    
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        prompt = (
            f"\n{self._prompt_prefix}\n"
            f"{request.summary()}\n"
            f"1. 仅允许这一次\n"
            f"2. 当前 Run 内允许完全相同的操作\n"
            f"3. 当前会话内允许完全相同的操作\n"
            f"4. 拒绝\n"
            f"请选择 [1/2/3/4]: "
        )
        answer = (await asyncio.to_thread(input, prompt)).strip()
        if answer == "1":
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVED,
                scope=ApprovalScope.ONCE,
            )
        if answer == "2":
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVED,
                scope=ApprovalScope.RUN,
            )
        if answer == "3":
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVED,
                scope=ApprovalScope.CONVERSATION,
            )
        return ApprovalResponse(decision=ApprovalDecision.DENIED)
