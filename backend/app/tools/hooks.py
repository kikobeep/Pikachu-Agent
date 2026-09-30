"""工具执行生命周期 Hook 的公共类型与调度器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tools.config import ToolExecutionContext, ToolResult
from .approval import ApprovalDecision, ApprovalRequest
from .permissions.models import PermissionRule


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
    """控制型 Hook 对本次工具执行给出的决定。"""

    denied_reason: str | None = None
    approval_request: ApprovalRequest | None = None
    matched_rule: PermissionRule | None = None


class ToolHook:
    """工具执行生命周期的异步 Hook。"""

    critical = False

    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        """在工具执行前调用；控制型 Hook 可以返回执行决定。"""

        return None

    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        """在等待人工审批前调用。"""

    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: PermissionRule | None = None,
    ) -> None:
        """在人工审批完成后调用；rule 表示本次审批创建或命中的规则。"""

    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        """工具产生统一结果后调用。"""


class ToolHookRunner:
    """依次调用 Hook，并按关键程度处理单个 Hook 的异常。"""

    def __init__(self, *hooks: ToolHook) -> None:
        self._hooks = hooks

    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        decision: ToolHookDecision | None = None
        for hook in self._hooks:
            try:
                current = await hook.before_execute(context)
                if current is not None and current.denied_reason is not None:
                    decision = current
                elif decision is None and current is not None:
                    decision = current
            except Exception as exc:
                if hook.critical:
                    return ToolHookDecision(
                        denied_reason=(
                            f"Critical tool hook failed: {type(exc).__name__}: {exc}"
                        )
                    )
                continue
        return decision

    async def on_approval_required(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.on_approval_required(context, request)
            except Exception:
                continue

    async def on_approval_completed(
        self,
        context: ToolExecutionContext,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        rule: PermissionRule | None = None,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.on_approval_completed(context, request, decision, rule)
            except Exception:
                continue

    async def after_execute(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        for hook in self._hooks:
            try:
                await hook.after_execute(context, result)
            except Exception:
                continue


__all__ = [
    "ToolExecutionContext",
    "ToolHook",
    "ToolHookDecision",
    "ToolHookRunner",
]
