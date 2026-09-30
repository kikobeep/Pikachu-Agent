from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from app.model.config import Message, MessageRole, ModelProvider, ModelRequest

from .summary import RollingConversationSummary, SummaryFact, SummaryGenerationResult

if TYPE_CHECKING:
    from app.model.adapter import ModelAdapter

_MAX_GOAL_CHARS = 160
_MAX_ENTRIES_PER_FIELD = 8
_MAX_ENTRY_CHARS = 80
_MAX_SUMMARY_CONTENT_CHARS = 1_600
_MAX_FACTS = 128
_SUMMARY_SYSTEM_PROMPT = """You maintain an evolving factual playbook for a long conversation.

Return only one JSON object. Do not output Markdown, explanations, or reasoning.

Output schema:
{
  "operations": [
    {
      "action": "ADD | UPDATE | MERGE | DROP",
      "id": "F001 or an existing fact id",
      "category": "person | preference | event | purchase | date | quantity | relationship | decision | other",
      "fact": "One atomic, explicit fact in English",
      "confidence": 0.0,
      "source": "short source hint such as messages 12-13",
      "merge_ids": ["existing duplicate ids; only for MERGE"]
    }
  ]
}

Rules:
- Extract only facts explicitly supported by the new messages. Never guess.
- Split different facts into separate entries; preserve exact names, dates, numbers, counts, prices, and ordering.
- Use ADD for a new fact, UPDATE when the same fact changed, MERGE for duplicates, and DROP only when a fact is explicitly invalidated or contradicted.
- For UPDATE, MERGE, and DROP, use the exact existing fact id from previous_summary.
- Do not delete an old fact merely because it is not mentioned in the new messages.
- Do not summarize the conversation into goals or vague topics.
- Return at most 32 operations; each fact must be concise but complete.
- All fact text must be in English.
"""


class ContextSummarizer(ABC):
    @abstractmethod
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message]
    ) -> SummaryGenerationResult:
        """将历史消息合并到已有摘要。"""


class ModelContextSummarizer(ContextSummarizer):
    def __init__(
        self,
        model_adapter: ModelAdapter,
        model_provider: ModelProvider,
        model: str | None = None,
        max_output_tokens: int = 1024,
    ):
        self.model_adapter = model_adapter
        self.model_provider = model_provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.provider_hint = model_provider.value
        self.model_hint = model or model_adapter.default_model

    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
    ) -> SummaryGenerationResult:
        return await self._summarize(
            previous_summary,
            messages,
        )

    async def retry_compact(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
        *,
        reason: str,
    ) -> SummaryGenerationResult:
        """首次摘要失败后的唯一重试入口；默认复用原摘要实现。"""

        return await self.summarize(previous_summary, messages)

    async def _summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message]
    ) -> SummaryGenerationResult:
        payload = {
            "previous_summary": (
                previous_summary.model_dump(mode="json")
                if previous_summary is not None
                else None
            ),
            "history_messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    "tool_calls": [call.model_dump(mode="json") for call in message.tool_calls],
                    "tool_call_id": message.tool_call_id,
                }
                for message in messages
            ],
            "schema": {
                "operations": "array of ADD, UPDATE, MERGE, DROP operations",
                "existing_fact_ids": [
                    fact.id
                    for fact in (previous_summary.facts if previous_summary else ())
                ],
            },
        }
        request = ModelRequest(
            messages=(
                Message(
                    role=MessageRole.SYSTEM,
                    content=_SUMMARY_SYSTEM_PROMPT
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(payload, ensure_ascii=False)

                )
            ),
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            extra_body=(
                {"thinking": {"type": "disabled"}}
                if self.model_provider == ModelProvider.DEEPSEEK else {}
            )
        )

        output = await self.model_adapter.complete(request=request)
        if output.finish_reason in ("length", "max_output_tokens", "incomplete", "failed"):
            raise ValueError("summary model response was incomplete")
        # ModelAdapter returns the unified ModelResponse shape.  The
        # assistant text is stored on ``message.content`` for both chat
        # completions and responses-style adapters; ``response`` was an old
        # pre-unification field and breaks DeepSeek summarization.
        content = output.message.content or ""
        if not content:
            raise ValueError("summary model returned empty content")
        result = _parse_json(content)
        summary = _curate_fact_operations(previous_summary, result)
        return SummaryGenerationResult(summary=summary, usage=output.usage)


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("summary model output must be a JSON object")

    return parsed


def _check_summary(summary: RollingConversationSummary) -> None:
    """对模型摘要执行硬限制，不能只依赖 Prompt 软约束。"""

    if summary.goal and len(summary.goal) > _MAX_GOAL_CHARS:
        raise ValueError(
            f"goal exceeds {_MAX_GOAL_CHARS} characters"
        )
    entry_fields = (
        "constraints",
        "done",
        "in_progress",
        "blocked",
        "key_decisions",
        "next_steps",
        "critical_context",
        "read_files",
        "modified_files",
    )
    total_chars = len(summary.goal or "")
    for field_name in entry_fields:
        entries = getattr(summary, field_name)
        if len(entries) > _MAX_ENTRIES_PER_FIELD:
            raise ValueError(
                f"{field_name} exceeds {_MAX_ENTRIES_PER_FIELD} entries"
            )
        for entry in entries:
            if len(entry) > _MAX_ENTRY_CHARS:
                raise ValueError(
                    f"{field_name} entry exceeds {_MAX_ENTRY_CHARS} characters"
                )
            total_chars += len(entry)
    if total_chars > _MAX_SUMMARY_CONTENT_CHARS:
        raise ValueError(
            f"summary content exceeds {_MAX_SUMMARY_CONTENT_CHARS} characters"
        )


def _curate_fact_operations(
    previous_summary: RollingConversationSummary | None,
    result: dict[str, Any],
) -> RollingConversationSummary:
    """Apply model-proposed deltas deterministically (ACE-style curator)."""

    facts: dict[str, SummaryFact] = {
        fact.id: fact
        for fact in (previous_summary.facts if previous_summary else ())
    }
    operations = result.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("summary operations must be an array")

    next_id = max(
        [
            int(fact_id[1:])
            for fact_id in facts
            if fact_id.startswith("F") and fact_id[1:].isdigit()
        ]
        or [0]
    ) + 1
    for raw in operations[:32]:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action", "")).upper().strip()
        fact_id = str(raw.get("id", "")).strip()
        if action == "ADD" and (not fact_id or fact_id in facts):
            while f"F{next_id:03d}" in facts:
                next_id += 1
            fact_id = f"F{next_id:03d}"
            next_id += 1
        if action == "DROP":
            if fact_id:
                facts.pop(fact_id, None)
            continue
        if action in {"UPDATE", "MERGE"} and fact_id not in facts:
            # A malformed update must not erase data; treat it as a new fact.
            action = "ADD"
            while f"F{next_id:03d}" in facts:
                next_id += 1
            fact_id = f"F{next_id:03d}"
            next_id += 1
        if action not in {"ADD", "UPDATE", "MERGE"}:
            continue
        fact_text = _normalize_text(str(raw.get("fact", "")))
        if not fact_text:
            continue
        facts[fact_id] = SummaryFact(
            id=fact_id,
            category=_normalize_text(str(raw.get("category", "other"))) or "other",
            fact=fact_text,
            confidence=float(raw.get("confidence", 1.0)),
            source=(
                _normalize_text(str(raw["source"]))
                if raw.get("source") is not None
                else None
            ),
        )
        if action == "MERGE":
            merge_ids = raw.get("merge_ids", [])
            if isinstance(merge_ids, list):
                for duplicate_id in merge_ids:
                    if duplicate_id != fact_id and isinstance(duplicate_id, str):
                        facts.pop(duplicate_id, None)

    ordered = tuple(facts.values())[-_MAX_FACTS:]
    return RollingConversationSummary(facts=ordered)


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


async def main() -> None:
    import argparse

    from app.model.adapter import ModelCompatibleAdapter
    from app.model.config import ModelConfig

    parser = argparse.ArgumentParser(description="调用模型生成示例会话摘要")
    parser.add_argument("--provider", choices=("openai", "qwen", "deepseek"),
                        help="默认使用 MODEL_DEFAULT_PROVIDER")
    args = parser.parse_args()

    settings = ModelConfig()
    config = settings.load_provider_config(args.provider or settings.model_default_provider)
    previous = RollingConversationSummary(
        goal="给个人记账工具增加账单导出功能",
        in_progress=("实现导出接口",),
        next_steps=("添加导出按钮",),
    )
    messages = (
            Message(
                role=MessageRole.USER,
                content="导出接口已经写好了，我用几条数据测过，CSV 能正常下载。",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="那接下来可以接页面按钮。导出范围是全部账单，还是按筛选条件？",
            ),
            Message(
                role=MessageRole.USER,
                content="按当前筛选条件。我一般按月份看账单，不想每次把全部记录都导出来。",
            ),
            Message(
                role=MessageRole.USER,
                content="另外我改主意了，先不做 CSV，改成 Excel，我爸妈用起来方便一点。",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="好的，需要把现有导出接口改成 Excel，并让按钮携带当前筛选条件。",
            ),
            Message(
                role=MessageRole.USER,
                content="对。金额保留两位小数，日期只要年月日。不要把备注导出，里面有私人信息。",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="是否需要支持一次导出多个账户？",
            ),
            Message(
                role=MessageRole.USER,
                content="暂时只导出当前账户。按钮放在账单列表右上角，今天先把接口改完，页面明天再做。",
            ),
        )

    adapter = ModelCompatibleAdapter(config)
    try:
        summarizer = ModelContextSummarizer(
            adapter, ModelProvider(config.provider), max_output_tokens=4096,
        )
        result = await summarizer.summarize(previous, messages)
        print(result.model_dump_json(indent=2))
    finally:
        await adapter.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
