"""会话模块的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.model.config import Message


@dataclass(frozen=True, slots=True)
class ConversationMessageRecord:
    """会话中的一条消息记录。"""

    sequence: int
    message: Message
    created_at: datetime
