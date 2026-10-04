"""Main Agent 单次 Run 的累计用量预算。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from app.model.config import ModelUsage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"


class RunBudgetStatus(StrEnum):
    """当前累计用量所处的预算阶段。"""

    DISABLED = "disabled"
    ACTIVE = "active"
    WARNING = "warning"
    FINALIZING = "finalizing"
    EXCEEDED = "exceeded"


class RunBudgetReason(StrEnum):
    """触发当前预算阶段的主指标。"""

    TOKENS = "tokens"
    MODEL_CALLS = "model_calls"


class RunBudgetConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="RUN_BUDGET_",
        extra="ignore",
    )

    enabled: bool = True
    warning_tokens: int = Field(default=80_000, ge=1)
    finalization_tokens: int = Field(default=120_000, ge=1)
    hard_tokens: int = Field(default=160_000, ge=1)
    # 默认只保留调用次数硬上限；旧配置仍可显式启用 Warning / Finalizing。
    warning_model_calls: int | None = Field(default=None, ge=1)
    finalization_model_calls: int | None = Field(default=None, ge=1)
    hard_model_calls: int = Field(default=20, ge=1)
    finalization_max_output_tokens: int = Field(default=1_200, ge=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> RunBudgetConfig:
        if not (
            self.warning_tokens
            < self.finalization_tokens
            < self.hard_tokens
        ):
            raise ValueError(
                "run budget token thresholds must satisfy warning < "
                "finalization < hard"
            )
        call_thresholds = [
            threshold
            for threshold in (
                self.warning_model_calls,
                self.finalization_model_calls,
                self.hard_model_calls,
            )
            if threshold is not None
        ]
        if any(
            left >= right
            for left, right in zip(call_thresholds, call_thresholds[1:])
        ):
            raise ValueError(
                "configured run budget model call thresholds must be "
                "strictly increasing"
            )
        return self


class RunBudgetDecision(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RunBudgetStatus
    reason: RunBudgetReason | None = None
    chargeable_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)

    @property
    def should_warn(self) -> bool:
        return self.status is RunBudgetStatus.WARNING

    @property
    def should_finalize(self) -> bool:
        return self.status is RunBudgetStatus.FINALIZING

    @property
    def exceeded(self) -> bool:
        return self.status is RunBudgetStatus.EXCEEDED


class RunBudget:
    """根据已确认的 Provider Usage 决定下一次主模型调用。"""

    def __init__(self, config: RunBudgetConfig | None = None) -> None:
        self.config = config or RunBudgetConfig()

    def evaluate(
        self,
        usage: ModelUsage,
        *,
        chargeable_tokens_override: int | None = None,
        model_calls_override: int | None = None,
    ) -> RunBudgetDecision:
        """评估累计用量；硬限制优先于收口和预警。"""

        chargeable = (
            chargeable_tokens(usage)
            if chargeable_tokens_override is None
            else chargeable_tokens_override
        )
        calls = (
            usage.model_calls
            if model_calls_override is None
            else model_calls_override
        )
        if not self.config.enabled:
            return RunBudgetDecision(
                status=RunBudgetStatus.DISABLED,
                chargeable_tokens=chargeable,
                model_calls=calls,
            )
        for status, token_limit, call_limit in (
            (
                RunBudgetStatus.EXCEEDED,
                self.config.hard_tokens,
                self.config.hard_model_calls,
            ),
            (
                RunBudgetStatus.FINALIZING,
                self.config.finalization_tokens,
                self.config.finalization_model_calls,
            ),
            (
                RunBudgetStatus.WARNING,
                self.config.warning_tokens,
                self.config.warning_model_calls,
            ),
        ):
            token_hit = chargeable >= token_limit
            call_hit = call_limit is not None and calls >= call_limit
            if token_hit or call_hit:
                return RunBudgetDecision(
                    status=status,
                    reason=(
                        RunBudgetReason.TOKENS
                        if token_hit
                        else RunBudgetReason.MODEL_CALLS
                    ),
                    chargeable_tokens=chargeable,
                    model_calls=calls,
                )
        return RunBudgetDecision(
            status=RunBudgetStatus.ACTIVE,
            chargeable_tokens=chargeable,
            model_calls=calls,
        )


def chargeable_tokens(usage: ModelUsage) -> int:
    """计算 V1 预算 Token：未缓存输入（未知则全部输入）加输出。"""

    input_tokens = (
        usage.uncached_input_tokens
        if usage.uncached_input_tokens is not None
        else usage.input_tokens
    )
    return input_tokens + usage.output_tokens


__all__ = [
    "RunBudget",
    "RunBudgetConfig",
    "RunBudgetDecision",
    "RunBudgetReason",
    "RunBudgetStatus",
    "chargeable_tokens",
]
