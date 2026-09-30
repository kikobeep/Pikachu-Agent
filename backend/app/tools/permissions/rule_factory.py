"""根据审批选择创建精确规则，并生成用户可读描述。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..approval import ApprovalRequest
from .models import ApprovalScope, PermissionEffect, PermissionRule


def build_safe_rule(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    scope: ApprovalScope,
    scope_id: str,
    effect: PermissionEffect = PermissionEffect.ALLOW,
) -> PermissionRule:
    """生成只匹配完整参数的规则，避免扩大 Shell 或 HTTP 权限。"""

    matcher_type = "tool_name" if tool_name == "web_search" else "exact_arguments"
    matcher = {} if matcher_type == "tool_name" else {"arguments": arguments}
    description = (
        f"允许使用 {tool_name}（忽略搜索参数）"
        if matcher_type == "tool_name"
        else f"允许调用 {tool_name}（完整参数相同）"
    )
    return PermissionRule(
        id=uuid4().hex,
        tool_name=tool_name,
        scope=scope,
        scope_id=scope_id,
        effect=effect,
        matcher_type=matcher_type,
        matcher=matcher,
        description=description,
        created_at=datetime.now(UTC),
    )


def describe_safe_rule(request: ApprovalRequest) -> str:
    """为审批菜单第 3 项生成用户可读的安全规则描述。"""

    if request.tool_name == "web_search":
        return "当前会话记住 web_search（后续搜索参数无需完全相同）"
    return "当前会话记住该操作（仅完整参数相同时自动通过）"


__all__ = ["build_safe_rule", "describe_safe_rule"]
