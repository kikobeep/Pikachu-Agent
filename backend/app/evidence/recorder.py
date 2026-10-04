"""把工具原始输出写入 Evidence Store。"""

from __future__ import annotations

from hashlib import sha256

from app.tools.config import ToolExecutionContext
from app.tools.output import (
    RecordedToolOutput,
    ToolOutputAttribution,
    ToolOutputAttributionResolver,
)

from .store import EvidenceStore


class EvidenceRecorder:
    """在模型预览截断前保存可复查的外部/工具事实。"""

    def __init__(
        self,
        store: EvidenceStore,
        *,
        attribution_resolver: ToolOutputAttributionResolver | None = None,
    ) -> None:
        self._store = store
        self._attribution_resolver = attribution_resolver

    async def record(
        self,
        context: ToolExecutionContext,
        content: str,
    ) -> RecordedToolOutput | None:
        if not context.run_id or not context.conversation_id:
            return None
        tool_name = context.tool_call.name
        definition = context.tool_definition
        if definition is not None and not definition.record_output:
            return None
        
        attribution = ToolOutputAttribution() # plan_id和plan_step 计划归属信息
        if self._attribution_resolver is not None:
            try:
                attribution = await self._attribution_resolver.resolve( # 查询plan_id和plan_step
                    context.conversation_id
                )
            except Exception:
                # 归属解析是尽力而为：失败时用空归属继续，不阻止证据写入。
                pass
        digest = sha256(content.encode("utf-8")).hexdigest()
        record = await self._store.create(
            conversation_id=context.conversation_id,
            run_id=context.run_id,
            tool_call_id=context.tool_call.id,
            tool_name=tool_name,
            content=content,
            sha256=digest,
            plan_id=attribution.plan_id,
            plan_step_id=attribution.plan_step_id,
        )
        return RecordedToolOutput(
            id=record.id,
            content_chars=record.content_chars,
            sha256=record.sha256,
        )
    
    


__all__ = [
    "EvidenceRecorder",
]
