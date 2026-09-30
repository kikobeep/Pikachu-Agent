"""上下文窗口与预算配置。

按模型族提供默认上下文窗口，并配置预算策略（安全余量、触发/目标比例）
与用户覆盖项。覆盖项（``context_window_override`` / ``max_output_tokens_override``）
作用于当前使用的模型（显式指定 ``context_override_model``，否则作用于当前
配置的默认模型），不会全局应用到所有模型。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"


class ContextSettings(BaseSettings):
    """上下文窗口与预算配置。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider 默认上下文窗口（用于 Provider 默认能力）
    context_window_default: int = Field(default=128_000, gt=0) # 
    context_window_openai: int = Field(default=200_000, gt=0)
    context_window_qwen: int = Field(default=1_000_000, gt=0)
    context_window_deepseek: int = Field(default=1_048_576, gt=0)
    context_window_anthropic: int = Field(default=200_000, gt=0)

    # Provider 默认最大输出（沿用原代码的数值）
    max_output_tokens_openai: int = Field(default=16_384, gt=0)
    max_output_tokens_qwen: int = Field(default=65_536, gt=0)
    max_output_tokens_deepseek: int = Field(default=393_216, gt=0)
    max_output_tokens_anthropic: int = Field(default=16_384, gt=0)
    
    # 模型窗口硬保护与日常工作预算
    '''
    允许输入上限 = 模型窗口 - 预留输出 - 安全空间
    日常输入预算 = min(允许输入上限, 64_000)

    压缩触发线 = min(
        允许输入上限 * 0.80,
        日常输入预算 * 0.80,
    )

    压缩目标 = min(
        允许输入上限 * 0.60,
        日常输入预算 * 0.45,
    )
    '''
    context_safety_margin_tokens: int = Field(default=4_096, ge=0)     # 额外预留的安全空间
    context_trigger_ratio: float = Field(default=0.80, gt=0.0, lt=1.0) # 按模型允许输入上限计算的压缩触发比例
    context_target_ratio: float = Field(default=0.60, gt=0.0, lt=1.0)  # 按模型允许输入上限计算的压缩目标比例
    context_preferred_input_tokens: int = Field(default=64_000, gt=0)  # 日常工作希望采用的输入预算
    context_working_trigger_ratio: float = Field(default=0.80, gt=0.0, le=1.0) # 日常工作预算的压缩触发比例
    context_working_target_ratio: float = Field(default=0.45, gt=0.0, lt=1.0) # 日常工作预算的压缩目标比例

    # 工具结果占用的空间
    context_tool_result_budget_ratio: float = Field(default=0.35, gt=0.0, lt=1.0) # 表示工具结果的预算按最终压缩目标的 35%
    context_max_tool_result_chars: int = Field(default=8_000, gt=0) # 单条工具结果过长时，触发文本截断的字符阈值
    context_tool_result_head_chars: int = Field(default=4_000, ge=0) # 截断时保留开头的字符数
    context_tool_result_tail_chars: int = Field(default=2_000, ge=0) # 截断时保留结尾的字符数

    # 历史压缩时保留多少近期内容
    context_keep_recent_tool_rounds: int = Field(default=2, ge=0) # 工具历史缩减时保护最近两轮工具调用
    context_keep_recent_conversation_blocks: int = Field(default=4, ge=0) # 生成历史摘要时保留最近四个对话块，优先总结更早的历史
    context_max_unsummarized_conversation_blocks: int = Field(default=30, gt=0) # 未摘要对话块过多时触发历史压缩，避免只依赖 token 阈值
    context_summary_max_output_tokens: int = Field(default=1_024, gt=0) # 生成摘要时，摘要模型允许输出的最大 token 数

    # 覆盖某个模型的容量配置
    context_override_provider: str | None = None
    context_override_model: str | None = None
    context_window_override: int | None = Field(default=None, gt=0)
    max_output_tokens_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_tool_result_segments(self) -> ContextSettings:
        """校验各层压缩目标与工具结果截断参数。"""

        retained = (
            self.context_tool_result_head_chars + self.context_tool_result_tail_chars
        )
        if retained > self.context_max_tool_result_chars:
            raise ValueError(
                "context tool result head/tail chars cannot exceed "
                "context_max_tool_result_chars"
            )
        if self.context_working_target_ratio >= self.context_working_trigger_ratio:
            raise ValueError(
                "context_working_target_ratio must be lower than "
                "context_working_trigger_ratio"
            )
        return self


class ContextSummaryModelConfig(BaseSettings):
    """滚动会话摘要使用的独立模型配置。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="CONTEXT_SUMMARY_",
        extra="ignore",
    )

    enabled: bool = True
    provider: str | None = None
    model: str | None = None

    @field_validator("provider", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("summary provider and model must be strings")
        return value.strip() or None


__all__ = ["ContextSettings", "ContextSummaryModelConfig"]
