from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from app.types import Message, MessageRole
from app.model.adapter import ModelAdapter
from app.model.type import ModelProvider, ModelRequest
from .summary import RollingConversationSummary, SummaryGenerationResult

from abc import ABC, abstractmethod
_MAX_OBJECTIVE_CHARS = 160
_PREFERRED_ENTRIES_PER_FIELD = 5
_MAX_ENTRIES_PER_FIELD = 8
_MAX_ENTRY_CHARS = 80
_MAX_SUMMARY_CONTENT_CHARS = 1_200
_SUMMARY_SYSTEM_PROMPT = """你是会话压缩器，把旧摘要和新增历史合并成紧凑结构化摘要。

要求：
- 只输出一个 JSON 对象，不要输出 Markdown、解释或任何思考过程；
- 只能保留输入中明确存在的信息，禁止补充、推断或编造事实；
- 只保留后续继续任务需要的信息，删除重复与冗余内容；
- 不要推测或复制外部 Task Snapshot 的步骤状态，Task 是独立事实源；
- 重要工具原文若带 evidence_id，只保留“用途 + 完整 evidence_id”引用，不复制
  大段原文，也不能缩写或修改 ID；后续可用 evidence_read 重新读取；
- 每个数组最多 5 条，每条不超过 80 个中文字符；
- 当前目标不超过 200 个字符，全部字段内容合计不超过 2000 个字符；
- 没有内容的字段使用 null 或空数组；
- 摘要必须明显短于输入历史。"""

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
        model_provider:ModelProvider,
        model: str | None = None,
        max_output_tokens: int = 1024,
    ):
        self.model_adapter = model_adapter
        self.model_provider = model_provider
        self.model = model
        self.max_output_tokens = max_output_tokens
    
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message],
    ) -> SummaryGenerationResult:
        return await self._summarize(
            previous_summary,
            messages,
        )
    
    async def _summarize(
        self,
        previous_summary: RollingConversationSummary | None,
        messages: Sequence[Message]
    ):
        payload = {
            "previous_summary":(
                previous_summary.model_dump(mode="json")
                if previous_summary is not None
                else None
            ),
            "history_messages":[
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in messages
            ],
            "schema": RollingConversationSummary.model_json_schema()
        }
        request = ModelRequest(
            messages = (
                Message(
                    role = MessageRole.SYSTEM,
                    content = _SUMMARY_SYSTEM_PROMPT
                ),
                Message(
                    role = MessageRole.USER,
                    content = json.dumps(payload,ensure_ascii=False)

                )
            ),
            model = self.model,
            max_output_tokens = self.max_output_tokens,
            extra_body=(
                {"thinking": {"type": "disabled"}}
                if self.model_provider == ModelProvider.DEEPSEEK else {}
            )
        )
       
        output = await self.model_adapter.complete(request = request)
        if output.finish_reason in ("length", "max_output_tokens", "incomplete", "failed"):
            raise ValueError("summary model response was incomplete")
        if not output.response.content:
            raise ValueError("summary model returned empty content")
        result = _parse_json(output.response.content)
        summary = RollingConversationSummary.model_validate(result)
        _check_summary(summary)
        return SummaryGenerationResult(summary=summary, usage=output.usage)
        
        


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    # 模型可能返回：```json
    #{"current_objective": "实现 CLI"}
    #```
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

    if summary.current_objective and len(summary.current_objective) > _MAX_OBJECTIVE_CHARS:
        raise ValueError(
            f"current_objective exceeds {_MAX_OBJECTIVE_CHARS} characters"
        )
    entry_fields = (
        "user_constraints",
        "key_decisions",
        "completed_work",
        "current_state",
        "pending_work",
        "important_facts",
    )
    total_chars = len(summary.current_objective or "")
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
        current_objective="给个人记账工具增加账单导出功能",
        pending_work=("实现导出接口", "添加导出按钮"),
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
