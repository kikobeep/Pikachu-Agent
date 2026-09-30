"""CLUSTER：把一批已完成 Run 聚类为重复任务模式。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

from app.model.config import Message, MessageRole, ModelRequest
from app.memory.utils import strip_json

from .models import MiningResult, TaskCluster
from .service import ImprovingSample

if TYPE_CHECKING:
    from app.model.adapter import ModelAdapter

logger = logging.getLogger("sidekick.skills_improve.miner")

_SYSTEM_PROMPT = """# 角色与职责
你是 Skill 改进管线的「模式挖掘」阶段。给定一批主 Agent 已完成的任务 Run
（每条含用户请求与工具执行证据），是否包含在本质上相似、可能反复出现、并且值得作为可复用流程来学习的任务，并把这类相似的多条 Run 归为一类。

## 输出纪律（必须遵守）
- 回复必须是一个 JSON 对象，且只能包含顶层字段 `clusters`。
- 不得返回空白、解释文字、Markdown 围栏、前后缀或第二个对象。
- 即使没有任何可靠模式，也必须返回精确的 `{"clusters": []}`，不能留空。
- `run_ids` 只能复制输入中的 `run_id`，每个 cluster 至少包含两个不同的输入 Run。

## 什么样的模式值得保留
1. 至少两个独立 Run 出现相同的操作和障碍，且步骤或约束结构相似。
2. 能被提炼成一条可复用的操作流程（步骤固定、边界清楚）。
3. 属于「怎么做一件事」的流程知识，而不是一次性的事实或项目决策。
4. 优先考虑具备以下特征的模式：多步骤工作流、反复出现的相似失败、用户反复纠正同一个错误、稳定的验证方法、明显可以避免的冗余步骤，以及在未来能显著降低的代价或错误率

## 什么不算模式
- 只出现一次、彼此无关的任务。
- 单纯的知识问答或一次性的临时操作
- 不要把简单的机械式单步操作蒸馏成技能，例如：重命名文件、读取文件、简单算术运算、任何单工具操作，或没有明显工作流的机械式操作。

## 聚类要求
- 每条 Run 至多归入一个 cluster；没有明显重复模式的 Run 可以不被归入任何 cluster。
- 优先比较结构化行为摘要：工具序列、失败工具、命令、自测、修改路径和终止原因。
- `outcome=passed` 和 `outcome=failed` 是外部评测结果；可以把同一模式下的成功与失败 Run 放在一起比较，但不要把失败过程本身当成成功步骤。
- `outcome` 绝不能作为拆分 cluster 的依据。同一个工作流即使同时有通过和失败 Run，也必须放进同一个 cluster，供后续 distiller 合并提炼成功做法与失败修复；只有操作目标或约束真正不同，才拆成多个 cluster。
- 对同一个可复用工作流只输出一个 cluster；不要分别输出“成功经验 cluster”和“失败经验 cluster”。
- 不要仅因为题目名称、编程语言或用户都说了“实现功能”就聚类。
- cluster 用短标题（title）概括「重复的是什么操作」，description 说明该模式。
- 宁可少而清晰，不要为了凑数制造模糊聚类。

## 输出要求
只返回一个合法 JSON 对象，不要添加解释或 Markdown 代码围栏。结构如下：

{
  "clusters": [
    {
      "id": "c1",
      "title": "给模块补充单元测试",
      "description": "反复出现：读取现有测试约定，为新增函数补齐 pytest 用例。",
      "run_ids": ["run_id_1", "run_id_2"]
    }
  ]
}

字段要求：
- `id`：cluster 的短标识（c1、c2 …）。
- `run_ids`：必须是输入里给出的 run id，且至少两个。
- 若没有任何值得保留的模式，返回 `{"clusters": []}`。
"""


class ModelPatternMiner:
    """用主模型把 ImprovingSample 聚类为 TaskCluster。"""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        model: str | None,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4_000,
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
    ) -> None:
        self._model_adapter = model_adapter
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts

    async def mine(self, samples: tuple[ImprovingSample, ...]) -> MiningResult:
        payload = self._render_samples(samples)
        model = self._model or self._model_adapter.default_model
        async with asyncio.timeout(self._timeout_seconds):
            for _attempt in range(1, self._max_attempts + 1):
                request_payload = payload
                if _attempt > 1:
                    request_payload += (
                        "\n\n上一轮没有返回可解析结果。请重试；只输出一个 JSON 对象，"
                        "没有模式时输出 {\"clusters\":[]}，不要输出任何解释。"
                    )
                request = ModelRequest(
                    messages=(
                        Message(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
                        Message(role=MessageRole.USER, content=request_payload),
                    ),
                    model=model,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    extra_body=_structured_output_options(self._model_adapter),
                )
                response = await self._model_adapter.complete(request)
                output = response.message.content or ""
                try:
                    data = json.loads(strip_json(output))
                    if not isinstance(data, dict):
                        raise ValueError("top-level JSON must be an object")
                    raw_clusters = data.get("clusters")
                    if not isinstance(raw_clusters, list):
                        raise ValueError("clusters must be a JSON array")
                    sample_ids = {sample.run.id for sample in samples}
                    clusters = tuple(
                        TaskCluster(
                            id=str(item.get("id", "")),
                            title=str(item.get("title", "")),
                            description=str(item.get("description", "")),
                            run_ids=tuple(
                                str(run_id)
                                for run_id in item.get("run_ids", ())
                                if isinstance(run_id, str) and run_id in sample_ids
                            ),
                        )
                        for item in raw_clusters
                        if isinstance(item, dict)
                        and len(
                            {
                                run_id
                                for run_id in item.get("run_ids", ())
                                if isinstance(run_id, str) and run_id in sample_ids
                            }
                        ) >= 2
                    )
                    return MiningResult(clusters=_merge_overlapping_clusters(clusters))
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
                    logger.warning(
                        "miner invalid output (attempt %s): %s; "
                        "finish_reason=%s content_len=%s reasoning_len=%s preview=%r",
                        _attempt,
                        exc,
                        response.finish_reason,
                        len(output),
                        len(response.message.reasoning or ""),
                        output[:300],
                    )
        raise ValueError("pattern miner returned invalid output")

    @staticmethod
    def _render_samples(samples: tuple[ImprovingSample, ...]) -> str:
        blocks = []
        for sample in samples:
            evidence = sample.evidence or ""
            events = sample.events
            tool_counts = Counter(
                event.tool_call.name
                for event in events
                if event.tool_call is not None
            )
            failed_tools = [
                event.tool_call.name
                for event in events
                if event.tool_call is not None
                and event.type.value == "tool_completed"
                and event.tool_result is not None
                and not event.tool_result.success
            ]
            tool_errors: list[str] = []
            command_outputs: list[str] = []
            for event in events:
                result = event.tool_result
                if result is None or result.success:
                    if (
                        result is not None
                        and result.output is not None
                        and event.tool_call is not None
                        and event.tool_call.name == "run_shell_command"
                    ):
                        output = str(result.output).replace("\n", " ")
                        command_outputs.append(output[:600])
                    continue
                detail = result.error or ""
                if result.output is not None:
                    output = str(result.output).replace("\n", " ")
                    detail = f"{detail} output={output}".strip()
                if detail:
                    tool_errors.append(
                        f"{result.tool_name}: {detail[:500]}"
                    )
            commands: list[str] = []
            paths: list[str] = []
            for event in events:
                if event.tool_call is None or not isinstance(event.tool_call.arguments, dict):
                    continue
                arguments = event.tool_call.arguments
                command = arguments.get("command")
                if isinstance(command, str) and command.strip():
                    commands.append(command.strip()[:240])
                for key in ("path", "file_path", "filename"):
                    value = arguments.get(key)
                    if isinstance(value, str) and value.strip():
                        paths.append(value.strip()[:200])
            trace_summary = {
                "run_status": sample.run.status.value,
                "stop_reason": sample.run.stop_reason,
                "tool_counts": dict(tool_counts),
                "failed_tools": failed_tools[:12],
                "tool_errors": tool_errors[-12:],
                "command_outputs": command_outputs[-12:],
                "commands": commands[-12:],
                "paths": paths[-30:],
                "skill_names": sorted(
                    {
                        event.skill_name
                        for event in events
                        if event.skill_name
                    }
                ),
                "event_types": dict(Counter(event.type.value for event in events)),
            }
            blocks.append(
                json.dumps(
                    {
                        "run_id": sample.run.id,
                        "outcome": sample.outcome,
                        "user_request": sample.user_request[:2_000],
                        "trace_summary": trace_summary,
                        "evidence": evidence[:6_000],
                    },
                    ensure_ascii=False,
                )
            )
        print("以下是一批已完成任务的 Run。请结合 trace_summary、用户请求和证据，找出重复的任务模式并聚类。\n\n"
            + "\n".join(blocks))
        return (
            "以下是一批已完成任务的 Run。请结合 trace_summary、用户请求和证据，找出重复的任务模式并聚类。\n\n"
            + "\n".join(blocks)
        )


def _structured_output_options(adapter: ModelAdapter) -> dict[str, object]:
    """Avoid spending the JSON miner budget on DeepSeek reasoning output."""

    if getattr(adapter, "provider", "") != "deepseek":
        return {}
    return {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def _merge_overlapping_clusters(
    clusters: tuple[TaskCluster, ...],
) -> tuple[TaskCluster, ...]:
    """Merge clusters that contain any of the same Runs.

    The miner contract assigns a Run to at most one cluster.  If the model
    returns a broad cluster plus narrower subsets, treating them separately
    would distill duplicate Skills.  A union preserves all evidence and gives
    distillation one candidate for that workflow.
    """

    merged: list[TaskCluster] = []
    for cluster in clusters:
        current_ids = set(cluster.run_ids)
        overlaps = [
            existing
            for existing in merged
            if current_ids.intersection(existing.run_ids)
        ]
        if not overlaps:
            merged.append(cluster)
            continue
        combined_ids = set(cluster.run_ids)
        titles = [cluster.title]
        descriptions = [cluster.description]
        for existing in overlaps:
            combined_ids.update(existing.run_ids)
            if existing.title:
                titles.append(existing.title)
            if existing.description:
                descriptions.append(existing.description)
            merged.remove(existing)
        merged.append(
            TaskCluster(
                id=overlaps[0].id,
                title=next((title for title in titles if title), ""),
                description="；".join(dict.fromkeys(descriptions)),
                run_ids=tuple(sorted(combined_ids)),
            )
        )
    return tuple(merged)
        


__all__ = ["ModelPatternMiner"]
