"""不可绕过的工具权限控制 Hook。

负责把权限档位、策略引擎匹配结果与人工审批门组合成最终决定：
- FORBIDDEN → 直接拒绝；
- HUMAN_APPROVAL → 先查策略引擎（规则命中则放行，规则拒绝则拒绝），
  否则交给人审批门；用户选择 RUN/CONVERSATION 时创建并保存规则。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.tools.config import ToolPermission

from .approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalScope,
)
from .config import ToolExecutionContext
from .hooks import ToolHook, ToolHookDecision
from .permissions.models import PermissionEffect, PermissionRule
from .permissions.policy import PermissionPolicyEngine
from .permissions.rule_factory import build_safe_rule
from .permissions.store import PermissionRuleStore

RuleFactory = Callable[..., PermissionRule]


@dataclass(frozen=True, slots=True)
class _ApprovalOutcome:
    """人工审批响应及其创建的可记忆规则。"""

    response: ApprovalResponse
    rule: PermissionRule | None = None


def _scope_ids(context: ToolExecutionContext) -> tuple[str, ...]:
    """规则评估涉及的作用域：Run 和当前会话。"""

    ids = (context.run_id, context.conversation_id)
    return tuple(scope_id for scope_id in ids if scope_id)


def _scope_id_for(scope: ApprovalScope, context: ToolExecutionContext) -> str | None:
    if scope is ApprovalScope.RUN:
        return context.run_id
    if scope is ApprovalScope.CONVERSATION:
        return context.conversation_id
    return None


class PermissionHook(ToolHook):
    """根据工具权限档位决定是否允许继续执行。"""

    critical = True

    def __init__(
        self,
        approval_gate: ApprovalGate,
        *,
        policy: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        rule_factory: RuleFactory = build_safe_rule,
    ) -> None:
        self._approval_gate = approval_gate
        self._policy = policy
        self._rule_store = rule_store
        self._rule_factory = rule_factory

    async def before_execute(
        self,
        context: ToolExecutionContext,
    ) -> ToolHookDecision | None:
        definition = context.tool_definition
        if definition is None:
            return None

        permission = definition.permission
        if permission is ToolPermission.FORBIDDEN:
            return ToolHookDecision(
                denied_reason=(
                    f"Tool '{context.tool_call.name}' is forbidden "
                    "for model execution."
                )
            )
        if permission is not ToolPermission.HUMAN_APPROVAL:
            return None

        # 需要人工审批时，先检查有没有适用的已保存权限规则，避免每次都重新询问用户

        request = ApprovalRequest(
            tool_call_id=context.tool_call.id,
            tool_name=context.tool_call.name,
            arguments=context.arguments or {},
            description=definition.description,
            run_id=context.run_id,
            conversation_id=context.conversation_id,
        )


        if self._policy is not None:
            verdict = await self._policy.evaluate(
                tool_name=context.tool_call.name,
                arguments=context.arguments or {},
                scope_ids=_scope_ids(context),
            )
            if verdict.effect is PermissionEffect.DENY:
                return ToolHookDecision(
                    denied_reason=(
                        f"Tool '{context.tool_call.name}' execution was denied "
                        "by a stored permission rule."
                    )
                )
            if verdict.effect is PermissionEffect.ALLOW and verdict.rule_id:
                return ToolHookDecision(
                    approval_request=request,
                    matched_rule=verdict.rule,
                )

        return ToolHookDecision(approval_request=request)

    async def request_approval(
        self,
        request: ApprovalRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> _ApprovalOutcome:
        """把审批请求交给审批门；选择临时或会话范围时保存规则。"""

        response = await self._approval_gate.request_approval(request)
        if (
            response.decision is ApprovalDecision.APPROVED
            and response.scope in (ApprovalScope.RUN, ApprovalScope.CONVERSATION)
            and context is not None
        ):
            rule = await self._create_rule(request, response.scope, context)
            if rule is not None:
                return _ApprovalOutcome(response=response, rule=rule)
        return _ApprovalOutcome(response=response)

    async def _create_rule(
        self,
        request: ApprovalRequest,
        scope: ApprovalScope,
        context: ToolExecutionContext,
    ) -> PermissionRule | None:
        if self._rule_store is None:
            return None
        scope_id = _scope_id_for(scope, context)
        if not scope_id:
            return None
        rule = self._rule_factory(
            tool_name=request.tool_name,
            arguments=request.arguments,
            scope=scope,
            scope_id=scope_id,
        )
        await self._rule_store.add(rule)
        return rule

    async def clear_run_rules(self, run_id: str) -> int:
        """Run 结束时清理其临时审批规则。"""

        if self._rule_store is None:
            return 0
        return await self._rule_store.remove_scope(ApprovalScope.RUN, run_id)

    @staticmethod
    def denied_reason(
        context: ToolExecutionContext,
        decision: ApprovalDecision,
    ) -> str | None:
        """审批未通过时返回拒绝原因。"""

        if decision is ApprovalDecision.APPROVED:
            return None
        return (
            f"Tool '{context.tool_call.name}' execution was denied "
            "(requires human approval)."
        )


__all__ = ["PermissionHook"]
