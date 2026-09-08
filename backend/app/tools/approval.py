"""人工审批请求与可替换的审批入口。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol


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


class ApprovalGate(Protocol):
    async def request(self, request: ApprovalRequest) -> bool:
        """只有明确返回 True 才表示本次调用获批。"""
        ...


class ConsoleApprovalGate:
    """命令行审批；完整展示参数，只接受 y/yes，输入结束视为拒绝。"""

    async def request(self, request: ApprovalRequest) -> bool:
        prompt = request.summary(max_arguments=None) + "\n允许执行？[y/N] "
        try:
            answer = await asyncio.to_thread(input, prompt)
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}
