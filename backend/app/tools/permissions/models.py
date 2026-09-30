"""审批规则的请求、效果与规则模型。

``ApprovalScope`` 与 ``ApprovalResponse`` 定义在 ``approval.py``，
这里从那里导入并重新导出，避免与审批门形成循环导入。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..approval import ApprovalDecision, ApprovalResponse, ApprovalScope


class PermissionEffect(StrEnum):
    """规则对命中的工具调用产生的效果。"""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionRule(BaseModel):
    """一条可复用的审批规则。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    tool_name: str
    scope: ApprovalScope
    scope_id: str
    effect: PermissionEffect = PermissionEffect.ALLOW
    matcher_type: str
    matcher: dict[str, Any]
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("scope")
    @classmethod
    def validate_persisted_scope(cls, value: ApprovalScope) -> ApprovalScope:
        if value is ApprovalScope.ONCE:
            raise ValueError("ONCE approval cannot be stored as a permission rule")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PermissionVerdict(BaseModel):
    """权限策略引擎对一次工具调用的评估结果。"""

    model_config = ConfigDict(extra="forbid")

    effect: PermissionEffect
    rule_id: str | None = None
    rule: PermissionRule | None = None


__all__ = [
    "ApprovalDecision",
    "ApprovalResponse",
    "ApprovalScope",
    "PermissionEffect",
    "PermissionRule",
    "PermissionVerdict",
]
