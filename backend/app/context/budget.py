from dataclasses import dataclass

from app.context.capability import ModelCapabilities

from .config import ContextSettings

DEFAULT_TRIGGER_RATIO = 0.80
DEFAULT_TARGET_RATIO = 0.60
DEFAULT_SAFETY_MARGIN_TOKENS = 4_096


@dataclass(frozen=True)
class ContextBudget:
    """一次模型调用的输入预算与压缩触发线。"""

    context_window: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    input_budget: int
    working_input_budget: int
    hard_trigger_tokens: int
    hard_target_tokens: int
    trigger_tokens: int
    target_tokens: int
    tool_result_budget_tokens: int


class ContextBudgetPolicy:
    """根据模型能力计算输入预算与压缩触发/目标线。"""

    def __init__(
        self,
        *,
        trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
        preferred_input_tokens: int = 64_000,
        working_trigger_ratio: float = 0.80,
        working_target_ratio: float = 0.45,
        tool_result_budget_ratio: float = 0.35,
    ) -> None:
        if not 0.0 < trigger_ratio < 1.0:
            raise ValueError("trigger_ratio must be in (0, 1)")
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio must be in (0, 1)")
        if target_ratio >= trigger_ratio:
            raise ValueError("target_ratio must be lower than trigger_ratio")
        if safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if preferred_input_tokens <= 0:
            raise ValueError("preferred_input_tokens must be greater than zero")
        if not 0.0 < working_trigger_ratio <= 1.0:
            raise ValueError("working_trigger_ratio must be in (0, 1]")
        if not 0.0 < working_target_ratio < working_trigger_ratio:
            raise ValueError(
                "working_target_ratio must be lower than working_trigger_ratio"
            )
        if not 0.0 < tool_result_budget_ratio < 1.0:
            raise ValueError("tool_result_budget_ratio must be in (0, 1)")
        self._trigger_ratio = trigger_ratio
        self._target_ratio = target_ratio
        self._safety_margin_tokens = safety_margin_tokens
        self._preferred_input_tokens = preferred_input_tokens
        self._working_trigger_ratio = working_trigger_ratio
        self._working_target_ratio = working_target_ratio
        self._tool_result_budget_ratio = tool_result_budget_ratio

    def compute(
        self,
        capabilities: ModelCapabilities,
        *,
        max_output_tokens: int | None = None,
    ) -> ContextBudget:
        """计算预算；本次显式 max_output_tokens 优先于模型默认值。"""

        context_window = capabilities.context_window
        reserved_output = (
            max_output_tokens
            if max_output_tokens is not None
            else capabilities.max_output_tokens
        )
        if reserved_output <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if reserved_output > capabilities.max_output_tokens:
            raise ValueError(
                f"max_output_tokens ({reserved_output}) exceed model maximum "
                f"({capabilities.max_output_tokens})"
            )

        input_budget = (
            context_window - reserved_output - self._safety_margin_tokens
        )
        if input_budget <= 0:
            raise ValueError(
                f"invalid context budget: window={context_window} "
                f"reserved_output={reserved_output} "
                f"safety_margin={self._safety_margin_tokens} "
                f"input_budget={input_budget} (must be > 0)"
            )
        working_input_budget = min(input_budget, self._preferred_input_tokens)
        hard_trigger_tokens = int(input_budget * self._trigger_ratio)
        hard_target_tokens = int(input_budget * self._target_ratio)
        trigger_tokens = min(
            hard_trigger_tokens,
            int(working_input_budget * self._working_trigger_ratio),
        )
        target_tokens = min(
            hard_target_tokens,
            int(working_input_budget * self._working_target_ratio),
        )
        return ContextBudget(
            context_window=context_window,
            reserved_output_tokens=reserved_output,
            safety_margin_tokens=self._safety_margin_tokens,
            input_budget=input_budget,
            working_input_budget=working_input_budget,
            hard_trigger_tokens=hard_trigger_tokens,
            hard_target_tokens=hard_target_tokens,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            tool_result_budget_tokens=int(
                target_tokens * self._tool_result_budget_ratio
            ),
        )


def build_budget_policy(
    settings: ContextSettings | None = None,
) -> ContextBudgetPolicy:
    """从配置构建预算策略。"""

    settings = settings or ContextSettings()
    return ContextBudgetPolicy(
        trigger_ratio=settings.context_trigger_ratio,
        target_ratio=settings.context_target_ratio,
        safety_margin_tokens=settings.context_safety_margin_tokens,
        preferred_input_tokens=settings.context_preferred_input_tokens,
        working_trigger_ratio=settings.context_working_trigger_ratio,
        working_target_ratio=settings.context_working_target_ratio,
        tool_result_budget_ratio=settings.context_tool_result_budget_ratio,
    )


__all__ = [
    "ContextBudget",
    "ContextBudgetPolicy",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_TARGET_RATIO",
    "DEFAULT_TRIGGER_RATIO",
    "build_budget_policy",
]
