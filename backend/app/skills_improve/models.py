"""Skill 自进化的数据模型（CLUSTER → DISTILL → Candidate）。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.skills.utils import validate_skill_name


class CandidateStatus(StrEnum):
    """Skill Candidate 的审核状态。"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TaskCluster(BaseModel):
    """被识别为同一重复任务模式的一组 Run。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str = ""
    description: str = ""
    run_ids: tuple[str, ...] = ()


class MiningResult(BaseModel):
    """PatternMiner 的一次聚类输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clusters: tuple[TaskCluster, ...] = ()


class SkillCandidate(BaseModel):
    """一次蒸馏产生的候选 Skill，待人工审核。

    ``content`` 只保存 SKILL.md 正文（不含 front matter）；审核通过时再拼接
    front matter 落盘，避免候选阶段就把元数据写死。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str
    content: str
    source_run_ids: tuple[str, ...] = ()
    cluster_id: str | None = None
    status: CandidateStatus = CandidateStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return validate_skill_name(value)


class DistillationResult(BaseModel):
    """ProcedureDistiller 对单个 cluster 的一次蒸馏输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SkillCandidate | None = None
    skip_reason: str | None = None


__all__ = [
    "CandidateStatus",
    "DistillationResult",
    "MiningResult",
    "SkillCandidate",
    "TaskCluster",
]
