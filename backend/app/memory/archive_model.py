from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from app.model.config import Message, MessageRole
from app.model.config import ModelProvider, ModelRequest, ModelUsage

from .archive_config import (
    ArchiveAction,
    MemoryArchiveConfig,
    MemoryArchiveDecision,
    MemoryArchiveInput,
    MemoryArchiveResponse,
)
from .utils import add_usage, strip_json

if TYPE_CHECKING:
    from app.model.adapter import ModelAdapter


_MEMORY_ARCHIVE_SYSTEM_PROMPT = """
# 角色与目标

你是长期记忆容量维护助手
当前活跃记忆已达到或超过容量上限。请评估提供的候选记忆，判断是否可以归档其中一条，以释放容量。

## 判断规则
仅评估候选列表中的记忆，每次最多选择一条归档。

以下情况可以考虑归档：

- 信息已经过时。
- 与其他记忆重复。
- 已被新的信息或决策取代。
- 不再具有跨会话使用价值。

如果候选记忆各自仍有独立价值，或缺乏充分的归档依据，应选择暂缓处理。

## 操作边界

- 只能选择 `archive`（归档）或 `defer`（暂缓）。
- 不得编造记忆 ID。
- 不得修改记忆内容或合并记忆。
- 不得回答用户。
- 归档会将 Markdown 文件移入可恢复的归档目录，不是删除。

## 输出格式

只返回一个合法的 JSON 对象，不要添加解释或 Markdown 代码围栏。

{
  "action": "defer",
  "memory_id": null,
  "reason": "简要说明判断依据"
}

字段要求：

- `action`：只能为 `archive` 或 `defer`。
- `memory_id`：选择 `archive` 时，必须是候选列表中的一个 ID；选择 `defer` 时，必须为 `null`。
- `reason`：说明归档或暂缓的依据。
"""


class ArchiveMemoryReflector:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        default_model: str,
        provider: ModelProvider,
        config: MemoryArchiveConfig,
    ):
        self._model_adapter = adapter
        self._config = config or MemoryArchiveConfig()
        self._default_model = default_model

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def provider_hint(self) -> str | None:
        return self._config.provider or ""

    @property
    def config(self) -> MemoryArchiveConfig:
        return self._config

    @property
    def model_hint(self) -> str:
        return (
            self._config.model
            or self._default_model
            or self._model_adapter.default_model
        )

    async def decide(
        self,
        archive_input: MemoryArchiveInput,
    ) -> MemoryArchiveResponse:
        if not self.enabled:
            return MemoryArchiveResponse()
        started = time.perf_counter()
        config = self._config
        adapter = self._model_adapter
        provider = adapter.provider

        if config.model is not None:
            model = config.model
        elif config.provider is not None:
            model = adapter.default_model
        else:
            model = self._default_model or adapter.default_model

        input_json = json.dumps(
            archive_input.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        request = ModelRequest(
            messages=(
                Message(
                    role=MessageRole.SYSTEM,
                    content=_MEMORY_ARCHIVE_SYSTEM_PROMPT,
                ),
                Message(
                    role=MessageRole.USER,
                    content=input_json,
                ),
            ),
            model=model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

        usage = None
        last_error = "archive model returned no decision"

        try:
            async with asyncio.timeout(config.timeout_seconds):
                for _ in range(config.max_attempts):
                    response = await adapter.complete(request)
                    usage = (
                        response.usage
                        if usage is None
                        else add_usage(usage, response.usage)
                    )

                    try:
                        output = response.message.content
                        if not output:
                            raise ValueError("archive model returned empty content")

                        decision = MemoryArchiveDecision.model_validate_json(
                            strip_json(output)
                        )
                        if (
                            decision.action is ArchiveAction.ARCHIVE
                            and decision.memory_id
                            not in {item.id for item in archive_input.candidates}
                        ):
                            raise ValueError(
                                "archive selected an ID outside the candidate set"
                            )
                    except ValueError as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        continue

                    return MemoryArchiveResponse(
                        decision=decision,
                        provider=provider,
                        model=model,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        usage=usage or ModelUsage(),
                    )

        except TimeoutError:
            last_error = (
                f"Archive generation timed out after {config.timeout_seconds:g} seconds"
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        return MemoryArchiveResponse(
            provider=provider,
            model=model,
            duration_ms=(time.perf_counter() - started) * 1000,
            usage=usage or ModelUsage(),
            error=last_error,
        )
