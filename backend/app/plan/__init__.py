"""plan 模块的公共接口。

工具和上下文提供器使用延迟导出，避免直接导入 ``app.plan.runner`` 时
提前加载 ``app.tools``，从而触发模型适配器与工具定义之间的循环导入。
"""

from .config import (
    PLAN_ID_LENGTH,
    Plan,
    PlanPatch,
    PlanPriority,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
)
from .store import (
    DEFAULT_PLANS_DIR,
    PlanStore,
)
from .runner import PlanRunner


def __getattr__(name: str):
    if name == "PlanContextProvider":
        from .context import PlanContextProvider

        return PlanContextProvider
    if name == "PlanToolOutputAttributionResolver":
        from .attribution import PlanToolOutputAttributionResolver

        return PlanToolOutputAttributionResolver
    if name in {
        "PlanCreateTool",
        "PlanGetTool",
        "PlanListTool",
        "PlanUpdateTool",
        "register_plan_tools",
    }:
        from .tools import (
            PlanCreateTool,
            PlanGetTool,
            PlanListTool,
            PlanUpdateTool,
            register_plan_tools,
        )

        return locals()[name]
    raise AttributeError(name)

__all__ = [
    'DEFAULT_PLANS_DIR',
    'PLAN_ID_LENGTH',
    'Plan',
    'PlanContextProvider',
    'PlanCreateTool',
    'PlanGetTool',
    'PlanListTool',
    'PlanPatch',
    'PlanPriority',
    'PlanRunner',
    'PlanStatus',
    'PlanStep',
    'PlanStepStatus',
    'PlanStore',
    'PlanToolOutputAttributionResolver',
    'PlanUpdateTool',
    'register_plan_tools',
]
