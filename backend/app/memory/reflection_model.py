from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from app.model.config import Message, MessageRole
from app.model.config import ModelRequest

from .reflection_config import (
    MemoryReflectionConfig,
    MemoryReflectionInput,
    MemoryReflectionReponse,
    ReflectionAction,
    ReflectionDecision,
)
from .utils import add_usage, strip_json

if TYPE_CHECKING:
    from app.model.adapter import ModelAdapter


_MEMORY_REFLECTION_SYSTEM_PROMPT = """
# 角色与职责
你是长期记忆整理助手，目前主 Agent 已完成本轮任务，你需要判断是否需要新增或更新普通长期记忆。
1. 每次最多提出一项记忆变更。普通记忆应保持精简，谨慎新增；
2. 存在明确且值得长期保留的变化时，也不要遗漏。

## 工作边界
主 Agent 已经完成用户的任务。你不得：

- 回答用户或继续执行任务。
- 调用工具或修改 Task。
- 修改核心记忆（Core Memory）。
- 创建技能（Skills）。

## 哪些内容值得保存
1. 普通长期记忆适合保存：
    - 重要的历史决策。
    - 用户的相关事实（如居住地、关系、爱好、未来计划等等）
    - 长期有效的方向变化。
    - 后续会话可能需要回顾的项目背景
    - 后续会话可能会用到的所有原子事实（比如日期、金钱、地点、人物、关系等等）

2. 不要保存：
    - 当前任务进度或待办步骤。
    - 临时约束或一次性信息。
    - 原始工具输出。
    - 可复用的操作流程。

### 核心记忆不属于本次处理范围
以下信息应由核心记忆保存，不得写入普通记忆：
    - 用户明确表达的稳定身份信息。
    - 真正适用于全局的长期偏好。
    - 全局性的安全或隐私约束。

如果本轮只有这类信息，应返回 `none`。

即使主 Agent 未调用 `core_memory_update`、未找到该按需工具，或核心记忆写入失败，也不得通过新增或更新普通记忆来补偿、兜底或备份。

## 如何判断是否需要变更

- 用户明确表示某项项目决策或规则已经确定、完成、纠正或扩展，可以作为长期记忆变更的依据。
- 已有记忆的内容与本次对话的内容有所冲突，已有记忆的内容已经过时。
- 本轮不必实际修改代码或文件。是否产生文件变更，与是否获得值得长期保留的项目知识，是两回事。
- 不得仅凭提议、推测或助手自身的说法，将某项决定视为已经确定。

## 如何选择操作
### `none`：不变更
没有值得长期保留的普通记忆变更时，返回 `none`。

### `create`：新增记忆
存在值得长期保留的信息，且不适合并入本轮已读取的记忆时，可以新增一条记忆。
避免将同一主题拆成多条重复或零散的记忆。

### `update`：更新记忆
只能更新 `memory_ids` 中列出的记忆。这些 ID 表明主 Agent 在本轮成功读取过对应记忆的完整正文。
如果判断某条已有记忆需要更新，但只有索引提示、没有读取过正文，应返回 `none`。不要猜测原文，也不要用不完整的信息覆盖原有细节。
如果新证据为已读取记忆的现有主题补充了长期有效的规则，应优先选择 `update`，而不是 `create` 或 `none`。新规则不必与旧内容矛盾，补充信息也可以构成更新。
更新时必须：
- 保留原记忆中仍然有效的事实。
- 保留当前证据中的重要否定条件和被否决的方案。
- 明确新旧决策或规则之间的替代关系。
- 保留相关数值限制及安全约束。
- 提供完整的替换内容，而不只是新增片段。

## 输出要求
只返回一个合法的 JSON 对象，不要添加解释或 Markdown 代码围栏。
结构如下，其中 `action` 只能为 `none`、`create` 或 `update`：

{
  "action": "none",
  "memory_id": null,
  "title": null,
  "summary": null,
  "content": null,
  "reason": "简要说明判断依据"
}

字段要求：
- `create`：必须提供 `title`、`summary` 和 `content`，`memory_id` 为 `null`。
- `update`：必须提供 `memory_id`，以及完整替换后的 `title`、`summary` 和 `content`，确保索引提示与正文一致。
- `none`：`memory_id`、`title`、`summary` 和 `content` 必须全部为 `null`。
- 所有操作都必须提供 `reason`。
"""


class MemoryReflector:
    def __init__(
        self,
        model_adapter: ModelAdapter,
        model: str | None,
        config: MemoryReflectionConfig,
    ):
        self._model_adapter = model_adapter
        self._model = model
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def provider_hint(self) -> str | None:
        return self._config.provider or getattr(self._model_adapter, "provider", None)

    @property
    def model_hint(self) -> str:
        return self._model or self._config.model or self._model_adapter.default_model

    async def decide(
        self, reflection_input: MemoryReflectionInput
    ) -> MemoryReflectionReponse:
        if not self.enabled:
            return MemoryReflectionReponse()
        started = time.perf_counter()
        input_json = json.dumps(
            reflection_input.model_dump(mode="json"), ensure_ascii=False
        )
        usage = None
        model = self._model or self._config.model or self._model_adapter.default_model
        async with asyncio.timeout(self._config.timeout_seconds):
            for attempt in range(1, self._config.max_attempts + 1):
                request = ModelRequest(
                    messages=(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_MEMORY_REFLECTION_SYSTEM_PROMPT,
                        ),
                        Message(role=MessageRole.USER, content=input_json),
                    ),
                    model=model,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_output_tokens,
                )
                response = await self._model_adapter.complete(request)
                output = response.message.content
                usage = (
                    response.usage
                    if usage is None
                    else add_usage(usage, response.usage)
                )
                try:
                    if not output:
                        raise ValueError("reflection model returned empty content")
                    decision = ReflectionDecision.model_validate_json(
                        strip_json(output)
                    )
                    if (
                        decision.action is ReflectionAction.UPDATE
                        and decision.memory_id not in reflection_input.memory_ids
                    ):
                        raise ValueError(
                            "reflection cannot update memory not recalled in this run"
                        )
                except (ValueError, TypeError) as exc:
                    if attempt < self._config.max_attempts:
                        continue
                    raise ValueError(
                        "reflection model returned invalid output"
                    ) from exc
                return MemoryReflectionReponse(
                    decision=decision,
                    provider=response.provider,
                    model=response.model,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    usage=usage,
                    attempts=attempt,
                    finish_reason=response.finish_reason,
                    input_json=input_json if self._config.capture_raw_io else None,
                    raw_output=output if self._config.capture_raw_io else None,
                )
        raise RuntimeError("reflection did not return a decision")
