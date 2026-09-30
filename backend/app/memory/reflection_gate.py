"""Post-Run Reflection 的保守确定性门控。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ReflectionGateReason(StrEnum):
    """只有确定性无长期价值的窄场景才允许跳过。"""

    SMALLTALK = "smalltalk"
    CAPABILITY_QUERY = "capability_query"
    EPHEMERAL_LOOKUP = "ephemeral_lookup"
    DURABLE_SIGNAL = "durable_signal"
    RECALLED_MEMORY = "recalled_memory"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ReflectionGateDecision:
    should_reflect: bool
    reason: ReflectionGateReason


_SMALLTALK = frozenset(
    {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "谢谢",
        "谢谢你",
        "好的",
        "好",
        "收到",
        "明白了",
        "辛苦了",
        "再见",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thankyou",
        "ok",
        "okay",
        "gotit",
        "bye",
    }
)
_DURABLE_MARKERS = (
    "以后",
    "从现在开始",
    "一直",
    "记住",
    "偏好",
    "我喜欢",
    "我不喜欢",
    "我住",
    "我的项目",
    "项目决定",
    "最终决定",
    "规则",
    "改为",
    "纠正",
    "长期",
    "remember",
    "fromnowon",
    "iprefer",
    "ialways",
    "inever",
    "ilive",
    "myproject",
    "wedecided",
    "projectdecision",
)
_CAPABILITY_NOUNS = ("mcp", "skill", "技能", "工具", "功能", "能力")
_CAPABILITY_QUESTIONS = (
    "有什么",
    "有哪些",
    "支持什么",
    "支持哪些",
    "能做什么",
    "可以做什么",
    "列出",
    "看看",
    "看一下",
    "看下",
    "what",
    "which",
    "list",
    "available",
    "support",
)
_EPHEMERAL_NOUNS = (
    "天气",
    "气温",
    "温度",
    "几点",
    "当前时间",
    "现在时间",
    "今天日期",
    "weather",
    "temperature",
    "forecast",
    "whattime",
    "currenttime",
    "todaydate",
)


def decide_reflection_gate(
    user_input: str,
    *,
    memory_ids: tuple[str, ...] = (),
) -> ReflectionGateDecision:
    """判断是否值得调用Reflector；不确定时一律保留模型判断权。"""

    normalized = _normalize(user_input)
    if memory_ids:
        return ReflectionGateDecision(True, ReflectionGateReason.RECALLED_MEMORY)
    # 输入包含长期信息的信号，允许反思
    if any(marker in normalized for marker in _DURABLE_MARKERS):
        return ReflectionGateDecision(True, ReflectionGateReason.DURABLE_SIGNAL)
    # 只是寒暄打招呼，跳过反思
    if normalized in _SMALLTALK:
        return ReflectionGateDecision(False, ReflectionGateReason.SMALLTALK)
    # 只是询问能力，跳过反思
    if any(noun in normalized for noun in _CAPABILITY_NOUNS) and any(
        question in normalized for question in _CAPABILITY_QUESTIONS
    ):
        return ReflectionGateDecision(
            False,
            ReflectionGateReason.CAPABILITY_QUERY,
        )
    # 属于临时信息查询，跳过反思
    if any(noun in normalized for noun in _EPHEMERAL_NOUNS):
        return ReflectionGateDecision(
            False,
            ReflectionGateReason.EPHEMERAL_LOOKUP,
        )
    return ReflectionGateDecision(True, ReflectionGateReason.UNCERTAIN)


def _normalize(value: str) -> str:
    # 把字符串里的空格、标点、特殊符号全部刷掉，并把所有字母变成小写
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


__all__ = [
    "ReflectionGateDecision",
    "ReflectionGateReason",
    "decide_reflection_gate",
]
