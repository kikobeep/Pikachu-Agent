"""Light 模式的状态协议和持久化模型。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LightStateType(StrEnum):
    """FlowState 论文中的四类状态；暂不启用 artifacts。"""

    PREFERENCES = "preferences"
    KNOWLEDGE = "knowledge"
    ATTRIBUTES = "attributes"
    JUDGEMENTS = "judgements"


class LightStateOperation(StrEnum):
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"


class LightStateRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relation: str
    target_state_id: str = Field(min_length=1)


class LightStateDelta(BaseModel):
    """模型提出的一条状态增量。"""

    model_config = ConfigDict(extra="ignore")

    operation: LightStateOperation
    state_id: str | None = None
    state_type: LightStateType | None = None
    summary: str | None = None
    value: dict[str, Any] | None = None
    evidence_ids: tuple[str, ...] = ()
    relations: tuple[LightStateRelation, ...] = ()


class LightToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class LightModelEnvelope(BaseModel):
    """主模型在 Light 模式下返回的四段式协议。"""

    model_config = ConfigDict(extra="ignore")

    state_focus: tuple[str, ...] = ()
    state_delta: tuple[LightStateDelta, ...] = ()
    tool_calls: tuple[LightToolCall, ...] = ()
    assistant_message: str | None = None


class LightStateNode(BaseModel):
    """数据库中的状态节点。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    state_type: LightStateType
    summary: str = Field(min_length=1)
    value: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    relations: tuple[LightStateRelation, ...] = ()
    source_run_id: str | None = None
    revision: int = Field(default=1, ge=1)
    status: str = "active"
    created_at: str
    updated_at: str
