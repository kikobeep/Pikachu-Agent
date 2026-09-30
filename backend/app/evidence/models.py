"""Evidence 模块的数据模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRecord(BaseModel):
    """Evidence 元数据记录（不含正文）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    conversation_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    content_type: str = "text/plain; charset=utf-8"
    content_chars: int = Field(ge=0)
    content_bytes: int = Field(ge=0)
    sha256: str
    task_id: str | None = None
    task_step_id: str | None = None
    created_at: datetime


class EvidenceDocument(BaseModel):
    """Evidence 正文及其元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: EvidenceRecord
    content: str


class EvidenceSearchHit(BaseModel):
    """Evidence 搜索结果片段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: EvidenceRecord
    snippet: str
