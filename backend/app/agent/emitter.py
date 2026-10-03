'''
Agent / Loop
    ↓ emitter.emit(...)
生成事件，交给 event_handler
    ├─ CLI Handler   → 在终端打印
    ├─ Trace Handler → 保存到数据库
    └─ 前端 Handler  → 通过 SSE / WebSocket 推送到页面
'''
from __future__ import annotations

import asyncio
from typing import Any

from .events import AgentEvent, AgentEventHandler, AgentEventType


class EventEmitter:
    """为单次运行补充公共标识、顺序并隔离处理器异常。"""

    def __init__(
        self,
        *,
        handler: AgentEventHandler,
        run_id: str,
        conversation_id: str | None,
    ) -> None:
        self._handler = handler
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._sequence = 0

    async def emit(
        self,
        event_type: AgentEventType,
        **payload: Any,
    ) -> None:
        event = AgentEvent(
            run_id=self._run_id,
            conversation_id=self._conversation_id,
            sequence=self._sequence,
            type=event_type,
            **payload,
        )
        self._sequence += 1
        try:
            await self._handler.handle(event)
        except Exception:
            return

    async def request(
        self,
        event_type: AgentEventType,
        **payload: Any,
    ) -> Any:
        """向事件处理器发起一次可返回结果的交互请求。"""

        event = AgentEvent(
            run_id=self._run_id,
            conversation_id=self._conversation_id,
            sequence=self._sequence,
            type=event_type,
            **payload,
        )
        self._sequence += 1
        try:
            return await self._handler.handle(event)
        except Exception:
            return None
