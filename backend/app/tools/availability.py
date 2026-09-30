"""工具在不同 Agent 模式下的能力边界。"""

from __future__ import annotations

from collections.abc import Collection

from app.model.config import AgentMode

PLAN_MODE_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "web_search",
        "get_current_time",
        "current_time",
        "memory_read",
        "memory_search",
        "history_search",
        "history_read",
        "evidence_search",
        "evidence_read",
        "task_create",
        "task_update",
        "task_get",
        "task_list",
    }
)


class ToolAvailabilityPolicy:
    """集中管理模式白名单和按需激活规则。"""

    def __init__(
        self,
        *,
        plan_allowed_tools: Collection[str] = PLAN_MODE_ALLOWED_TOOLS,
    ) -> None:
        self._plan_allowed_tools = frozenset(plan_allowed_tools)

    def allowed_tool_names(
        self,
        mode: AgentMode,
        *,
        registered_names: Collection[str],
    ) -> frozenset[str]:
        if mode is AgentMode.PLAN:
            return self._plan_allowed_tools.intersection(registered_names)
        return frozenset(registered_names)

    def is_activation_satisfied(
        self,
        name: str,
        mode: AgentMode,
        *,
        on_demand_names: Collection[str],
        activated_names: Collection[str],
    ) -> bool:
        """仅检查激活条件；模式限制由 allowed_tool_names 处理。"""
       
        if name not in on_demand_names or name in activated_names:
            return True
        return mode is AgentMode.PLAN and name in self._plan_allowed_tools


__all__ = ["PLAN_MODE_ALLOWED_TOOLS", "ToolAvailabilityPolicy"]
