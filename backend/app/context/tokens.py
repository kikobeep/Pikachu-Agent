"""基于 tiktoken 的 token 数量估算。

用于在每次模型调用前估算请求占用的 token 数（消息 + 工具定义），
作为上下文窗口管理的第一步。

精度策略：
- OpenAI 模型：优先使用 tiktoken 对应编码，不加系数。
- 非 OpenAI 模型（Qwen / DeepSeek / Anthropic 等）：tiktoken 没有它们的
  词表，统一用 cl100k_base 近似，并乘以 ``>1`` 的保守系数向上取整，
  避免低估导致上下文溢出。系数可按模型族覆盖。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import tiktoken
from app.model.config import Message
from app.context.blocks import ToolRoundBlock, partition_messages
from app.tools.config import ToolDefinition
from tiktoken import Encoding

DEFAULT_ENCODING = "cl100k_base"

# 非 OpenAI 模型的保守系数：这些模型的 BPE 与 cl100k_base 有差异，
# 用 >1 的系数向上取整，避免低估。
DEFAULT_FAMILY_FACTORS: dict[str, float] = {
    "openai": 1.0,
    "qwen": 1.2,
    "deepseek": 1.2,
    "anthropic": 1.15,
    "other": 1.25,
}


class TokenEstimator:
    """估算文本、消息序列与工具定义的 token 数量。"""

    def __init__(
        self,
        default_encoding: str = DEFAULT_ENCODING,
        factors: dict[str, float] | None = None,
    ) -> None:
        self._default_encoding = default_encoding
        self._factors = {**DEFAULT_FAMILY_FACTORS, **(factors or {})}
        self._cache: dict[str, Encoding] = {}

    def estimate_text(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        if not text:
            return 0
        base = len(self._encoding(model).encode(text, disallowed_special=()))
        factor = self.factor_for(provider, model)
        if factor <= 1.0:
            return base
        return math.ceil(base * factor)

    def estimate_messages(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        '''
        消息结构开销
        + 角色
        + 正文
        + 可选名称
        + 可选工具调用 ID
        + 每个工具调用的结构、名称和参数
        = 消息 token 估算总数
        '''
        total = 0
        for message in messages:
            # 每条消息的角色 / 格式开销
            total += 3
            if message.role is not None:
                total += self.estimate_text(
                    message.role.value,
                    model=model,
                    provider=provider,
                )
            if message.content:
                total += self.estimate_text(
                    message.content,
                    model=model,
                    provider=provider,
                )
            if message.name:
                total += 1 + self.estimate_text(
                    message.name,
                    model=model,
                    provider=provider,
                )
            if message.tool_call_id:
                total += 1 + self.estimate_text(
                    message.tool_call_id,
                    model=model,
                    provider=provider,
                )
            for call in message.tool_calls:
                total += 4  # 工具调用块开销
                total += self.estimate_text(
                    call.name,
                    model=model,
                    provider=provider,
                )
                arguments: Any = call.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                total += self.estimate_text(
                    arguments,
                    model=model,
                    provider=provider,
                )
        return total

    def estimate_tools(
        self,
        tools: Sequence[ToolDefinition],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        total = 0
        for tool in tools:
            total += 5  # 工具定义结构开销
            total += self.estimate_text(
                tool.name,
                model=model,
                provider=provider,
            )
            total += self.estimate_text(
                tool.description,
                model=model,
                provider=provider,
            )
            total += self.estimate_text(
                json.dumps(
                    tool.parameters,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                model=model,
                provider=provider,
            )
        return total

    def estimate_request(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        """估算一次完整模型请求（消息 + 工具定义）的 token 数。"""

        return self.estimate_messages(
            messages,
            model=model,
            provider=provider,
        ) + self.estimate_tools(tools, model=model, provider=provider)

    def estimate_tool_results(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> int:
        results = []
        for block in partition_messages(messages):
            if not isinstance(block, ToolRoundBlock):
                continue

            for message in block.messages[1:]:
                results.append(message)

        return self.estimate_messages(
            tuple(results),
            model=model,
            provider=provider,
        )


    def factor_for(self, provider: str | None, model: str | None) -> float:
        """返回指定模型族适用的保守系数。"""

        family = model_family(provider, model)
        return self._factors.get(family, self._factors["other"])

    def _encoding(self, model: str | None) -> Encoding:
        cache_key = model or self._default_encoding
        if cache_key not in self._cache:
            self._cache[cache_key] = self._encoding_for(model)
        return self._cache[cache_key]

    def _encoding_for(self, model: str | None) -> Encoding:
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass
        return tiktoken.get_encoding(self._default_encoding)


def model_family(provider: str | None, model: str | None) -> str:
    """识别模型族（openai / qwen / deepseek / anthropic / other）。

    优先按 provider 判断，其次按模型名判断。
    """

    if provider:
        name = provider.strip().lower()
        if name == "openai":
            return "openai"
        if name == "qwen":
            return "qwen"
        if name == "deepseek":
            return "deepseek"
        if name == "anthropic":
            return "anthropic"
    lowered = (model or "").lower()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in lowered:
        return "openai"
    if "qwen" in lowered:
        return "qwen"
    if "deepseek" in lowered:
        return "deepseek"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    return "other"


_DEFAULT = TokenEstimator()


def default_token_estimator() -> TokenEstimator:
    """返回进程内共享的默认估算器。"""

    return _DEFAULT


__all__ = [
    "DEFAULT_ENCODING",
    "DEFAULT_FAMILY_FACTORS",
    "TokenEstimator",
    "default_token_estimator",
    "model_family",
]
