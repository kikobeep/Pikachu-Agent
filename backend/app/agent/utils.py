
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.agent.result import ToolCallRecord
from app.context.summary import ConversationSummaryState
from app.memory.utils import normalize_memory_id
from app.model.config import Message, ModelUsage
from app.tools.config import ToolDefinition

from .budget import RunBudgetConfig, RunBudgetDecision, RunBudgetStatus
from app.model.config import ToolCall,ModelProvider


def get_memory_versions(records: tuple[ToolCallRecord, ...]):
    result: dict[str, int] = {}
    for record in records:
        if record.tool_call.name != "memory_read" or not record.result.success:
            continue
        arguments = record.tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        memory_id = arguments.get("memory_id")
        if not isinstance(memory_id, str):
            continue
        try:
            normalized = normalize_memory_id(memory_id)
        except ValueError:
            continue
        output = record.result.output
        revision = output.get("revision") if isinstance(output, dict) else None
        # 业务逻辑
        if (
            isinstance(output, dict)
            and output.get("found") is True
            and output.get("id") == normalized
            and type(revision) is int
            and revision > 0
        ):
            result[normalized] = revision
    return result



def skill_read_outcome(output: object) -> tuple[str | None, bool]:
    """解析 skill_read 结果，保留“查询成功但未找到”的失败语义。"""

    if isinstance(output, str):
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            return None, False
    elif isinstance(output, dict):
        payload = output
    else:
        return None, False
    name = payload.get("name")
    normalized_name = name if isinstance(name, str) and name else None
    return normalized_name, payload.get("found") is True

def run_budget_event_fields(
    decision: RunBudgetDecision,
    config: RunBudgetConfig,
    *,
    status: RunBudgetStatus | None = None,
) -> dict[str, object]:
    """把预算快照转换为 AgentEvent 的统一字段。"""

    return {
        "run_budget_status": (status or decision.status).value,
        "run_budget_reason": (
            decision.reason.value if decision.reason is not None else None
        ),
        "run_budget_chargeable_tokens": decision.chargeable_tokens,
        "run_budget_model_calls": decision.model_calls,
        "run_budget_warning_tokens": config.warning_tokens, # 达到多少 token 时发出警告
        "run_budget_finalization_tokens": config.finalization_tokens, # 达到多少 token 时要求开始收尾
        "run_budget_hard_tokens": config.hard_tokens, # token 硬上限
        "run_budget_warning_model_calls": config.warning_model_calls, # 达到多少次模型调用时发出警告
        "run_budget_finalization_model_calls": config.finalization_model_calls, # 达到多少次模型调用时要求开始收尾
        "run_budget_hard_model_calls": config.hard_model_calls,
    }

def run_budget_detail(decision: RunBudgetDecision) -> str:

    reason = decision.reason.value if decision.reason is not None else "unknown"
    return (
        f"reason={reason}, chargeable_tokens={decision.chargeable_tokens}, "
        f"model_calls={decision.model_calls}"
    )

def add_model_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    """聚合 Usage，并保留缓存字段的“未知”语义。"""

    left_has_usage = _has_model_usage(left)
    right_has_usage = _has_model_usage(right)

    def add_optional(left_value: int | None, right_value: int | None) -> int | None:
        if not left_has_usage:
            return right_value
        if not right_has_usage:
            return left_value
        if left_value is None or right_value is None:
            return None
        return left_value + right_value

    return ModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_input_tokens=add_optional(
            left.cached_input_tokens,
            right.cached_input_tokens,
        ),
        uncached_input_tokens=add_optional(
            left.uncached_input_tokens,
            right.uncached_input_tokens,
        ),
        cache_read_input_tokens=add_optional(
            left.cache_read_input_tokens,
            right.cache_read_input_tokens,
        ),
        cache_write_input_tokens=add_optional(
            left.cache_write_input_tokens,
            right.cache_write_input_tokens,
        ),
        model_calls=left.model_calls + right.model_calls,
    )


def _has_model_usage(usage: ModelUsage) -> bool:
    return bool(
        usage.input_tokens
        or usage.output_tokens
        or usage.total_tokens
        or usage.cached_input_tokens is not None
        or usage.uncached_input_tokens is not None
        or usage.cache_read_input_tokens is not None
        or usage.cache_write_input_tokens is not None
        or usage.model_calls
    )

def add_usage(total: ModelUsage, current: ModelUsage) -> ModelUsage:
    """累加多轮模型调用的 Token 用量。"""

    return add_model_usage(total, current)


def usage_call_count(usage: ModelUsage) -> int:

    if usage.model_calls > 0:
        return usage.model_calls
    if usage.input_tokens or usage.output_tokens or usage.total_tokens:
        return 1
    return 0


def offset_summary_state(
    state: ConversationSummaryState | None,
    offset: int,
) -> ConversationSummaryState | None:

    if state is None or offset == 0:
        return state
    covered_message_count = state.covered_message_count + offset
    if covered_message_count < 0:
        raise ValueError("summary offset moved covered_message_count below zero")
    return state.model_copy(
        update={"covered_message_count": covered_message_count}
    )

def is_tool_call_text(content: str | None) -> bool:
    """识别被模型错误输出为普通文本的常见工具协议标记。"""

    if not content:
        return False
    lowered = content.lower()
    return any(
        marker in lowered
        for marker in (
            "<tool_calls",
            "<｜dsml｜tool_calls",  # 兼容deepseek
            "<｜dsml｜invoke"
        )
    )


def tool_call_signature(tool_call: ToolCall) -> str:
    """为重复工具调用检测生成稳定签名。"""

    arguments: Any = tool_call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{tool_call.name}:{arguments}"

    canonical_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{tool_call.name}:{canonical_arguments}"


def plan_task_id_from_output(output: object) -> str | None:
    """从 task_create / task_update 的工具输出 JSON 中提取任务 ID。"""

    if isinstance(output, str):
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            return None
    elif isinstance(output, dict):
        payload = output
    else:
        return None
    if not isinstance(payload, dict):
        return None
    task_id = payload.get("id")
    return task_id if isinstance(task_id, str) and task_id else None


def provider_name(provider: ModelProvider | str | None) -> str | None:
    """把 Provider 枚举转换为事件可序列化名称。"""

    if isinstance(provider, ModelProvider):
        return provider.value
    return provider

@dataclass(frozen=True)
class RequestPrefixState:
    """保存当前 Run 最近一次已发送请求的稳定前缀。"""

    source_messages: tuple[Message, ...]
    context_messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    sent_messages: tuple[Message, ...]

    def extend(
        self,
        *,
        source_messages: tuple[Message, ...],
        context_messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> tuple[Message, ...] | None:
       
        if context_messages != self.context_messages or tools != self.tools:
            return None
        previous_count = len(self.source_messages)
        if len(source_messages) < previous_count:
            return None
        if source_messages[:previous_count] != self.source_messages:
            return None
        return (*self.sent_messages, *source_messages[previous_count:])

