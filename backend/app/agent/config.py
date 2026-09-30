
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_RUN_BUDGET_FINALIZATION_MESSAGE = (
    "本 Run 已达到 Main Agent 用量收口线。不要再调用工具，请基于已有证据立即"
    "给出简洁的最终答复；明确区分已完成、未完成和无法验证的内容，不要伪造"
    "执行结果。"
)
_RUN_BUDGET_CLOSING_MESSAGE = (
    "本 Run 已达到 Main Agent 用量收口线，现在进入 Closing。仅保留完成当前"
    "目标所必需的交付工具；不要继续搜索、调查或扩展任务。若已有结果尚未写入"
    "文件、发布为交付物或同步到任务状态，请立即完成这一次交付；否则直接给出"
    "简洁的最终答复。"
)
_RUN_BUDGET_WARNING_MESSAGE = (
    "本 Run 的累计 Main Agent 用量已进入预警区。请减少不必要的重复调查和工具"
    "调用，优先完成当前目标；仍可在确有必要时继续使用工具。"
)
_TOOL_ROUND_LIMIT_FALLBACK_MESSAGE = (
    "已达到本 Run 的工具调用轮次上限，系统已停止继续执行工具。已有工具结果"
    "仍保留在本轮记录中，但模型未能在无工具模式下生成可靠总结；如需继续，"
    "请基于当前结果提出下一步要求。"
)
_EMPTY_FINAL_RETRY_MESSAGE = (
    "上一条模型响应没有可展示文本，也没有工具调用。请基于已有上下文给出一条"
    "完整、可直接展示给用户的最终回答；不要只输出内部思考。"
)
_TEXTUAL_TOOL_CALL_RETRY_MESSAGE = (
    "上一条模型响应把工具调用协议作为普通文本输出，系统不会执行这类文本。"
    "如需调用工具，必须使用 Provider 的结构化 tool_calls；否则请直接给出一条"
    "完整、可展示给用户的最终回答，不要输出 DSML、XML 或其他工具协议标记。"
)

