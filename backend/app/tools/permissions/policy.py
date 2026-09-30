"""权限策略引擎：匹配已存规则，返回 ALLOW / ASK / DENY。

引擎只负责“规则匹配”，不负责与用户交互（那是 ApprovalGate 的事），
也不负责规则的创建（那是 rule_factory + store 的事）。
"""

from __future__ import annotations

from collections.abc import Sequence

from .matchers import build_matcher
from .models import ApprovalScope, PermissionEffect, PermissionRule, PermissionVerdict
from .store import PermissionRuleStore


class PermissionPolicyEngine:
    """在给定作用域内查找与当前调用匹配的规则。"""

    def __init__(self, store: PermissionRuleStore) -> None:
        self._store = store

    async def evaluate(
        self,
        *,
        tool_name: str,
        arguments: dict,
        scope_ids: Sequence[str],
    ) -> PermissionVerdict:
    
        matched_rules: list[PermissionRule] = []
        rules = await self._store.list(scope_ids=tuple(scope_ids))
        for rule in rules:
            if rule.tool_name != tool_name:
                continue
            try:
                matcher = build_matcher(rule.matcher_type, rule.matcher)
            except ValueError:
                continue
            try:
                matched = matcher.matches(arguments)
            except Exception:
                continue
            if matched:
                matched_rules.append(rule)
        if matched_rules:
            selected = max(matched_rules, key=_rule_priority)
            return PermissionVerdict(
                effect=selected.effect,
                rule_id=selected.id,
                rule=selected,
            )
        return PermissionVerdict(effect=PermissionEffect.ASK)

    @property
    def store(self) -> PermissionRuleStore:
        """返回策略使用的规则存储，供执行器校验依赖一致性。"""

        return self._store


def _rule_priority(rule: PermissionRule) -> tuple[int, int, float]:
    """规则冲突时优先 DENY，其次更具体的 Run，最后选择较新规则。"""

    effect_priority = {
        PermissionEffect.ALLOW: 1,
        PermissionEffect.ASK: 2,
        PermissionEffect.DENY: 3,
    }[rule.effect]
    scope_priority = 2 if rule.scope is ApprovalScope.RUN else 1
    return effect_priority, scope_priority, rule.created_at.timestamp()


__all__ = ["PermissionPolicyEngine"]
