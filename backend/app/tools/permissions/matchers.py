"""审批规则的匹配器：判断一次工具调用是否命中某条规则。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class PermissionMatcher(ABC):
    """判断一次工具调用是否命中某条规则。"""

    @abstractmethod
    def matches(self, arguments: dict[str, Any]) -> bool:
        """返回是否命中。"""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class ExactArgumentsMatcher(PermissionMatcher):
    """参数与已批准操作完全相同。"""

    def __init__(self, arguments: dict[str, Any]) -> None:
        self._expected = _canonical(arguments or {})

    def matches(self, arguments: dict[str, Any]) -> bool:
        return self._expected == _canonical(arguments or {})


class ToolNameMatcher(PermissionMatcher):
    """Match any arguments for a tool whose name already matched in policy."""

    def __init__(self) -> None:
        pass

    def matches(self, arguments: dict[str, Any]) -> bool:
        return True


_MATCHERS: dict[str, type[PermissionMatcher]] = {
    "exact_arguments": ExactArgumentsMatcher,
    "tool_name": ToolNameMatcher,
}


def build_matcher(
    matcher_type: str,
    payload: dict[str, Any],
) -> PermissionMatcher:
    """按 matcher_type 构造匹配器。"""

    matcher_cls = _MATCHERS.get(matcher_type)
    if matcher_cls is None:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
    return matcher_cls(**payload)


__all__ = [
    "ExactArgumentsMatcher",
    "ToolNameMatcher",
    "PermissionMatcher",
    "build_matcher",
]
