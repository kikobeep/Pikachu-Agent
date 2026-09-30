
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from app.model.config import Message, MessageRole
from .prompt import MEMORY_SEARCH_HEADER

DEFAULT_SEARCH_DATABASE_NAME = "memorysearch.sqlite"

class SearchMode(StrEnum):
    """一次检索实际使用的融合模式。"""

    HYBRID = "hybrid"
    VECTOR = "vector"
    BM25 = "bm25"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class MemorySearchCandidate:
    """一个按 memory_id 合并后的检索候选。"""

    memory_id: str
    title: str
    summary: str
    revision: int
    snippet: str
    rrf_score: float
    vector_similarity: float | None
    matched_by_vector: bool
    matched_by_text: bool
    vector_rank: int | None = None
    text_rank: int | None = None


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """一次检索的完整结果（含降级信息）。"""

    mode: SearchMode
    candidates: tuple[MemorySearchCandidate, ...]
    query: str
    degrade_reason: str | None = None

    def render_message(self, *, max_chars: int = 2_400) -> Message | None:
        """渲染为临时注入的系统消息；无候选时返回 None（不注入噪声）。"""

        if not self.candidates:
            return None
        lines = [MEMORY_SEARCH_HEADER.rstrip()]
        used = len(lines[0])
        for candidate in self.candidates:
            entry_lines = [
                f"[{candidate.memory_id}] {candidate.title} "
                f"(revision {candidate.revision})",
                f"Summary: {candidate.summary}",
            ]
            if candidate.snippet:
                entry_lines.append(f"Snippet: {candidate.snippet}")
            entry = "\n".join(entry_lines)
            if used + len(entry) > max_chars:
                break
            lines.append("")
            lines.append(entry)
            used += len(entry) + 1
        if len(lines) == 1:
            return None
        return Message(
            role=MessageRole.SYSTEM,
            # name=MEMORY_RECALL_MESSAGE_NAME,
            content="\n".join(lines).rstrip() + "\n",
        )

class MemorySearchConfig:
    """检索与召回的运行参数（纯代码常量，不引入额外环境变量面）。"""

    def __init__(
        self,
        *,
        top_k: int = 5,
        vector_top_k: int = 1,
        text_top_k: int = 1,
        vector_only: bool = False,
        chunk_chars: int = 900,
        chunk_overlap_chars: int = 180,
        max_chunks_per_memory: int = 16,
        candidate_multiplier: int = 8,
        min_vector_similarity: float = 0.12,
        snippet_chars: int = 360,
        recall_message_max_chars: int = 2_400,
        query_max_chars: int = 1_600,
    ) -> None:
        if top_k <= 0 or chunk_chars <= 0:
            raise ValueError("search settings limits must be positive")
        if vector_top_k <= 0 or text_top_k <= 0:
            raise ValueError("per-source search limits must be positive")
        if not 0 <= chunk_overlap_chars < chunk_chars:
            raise ValueError("chunk overlap must be within [0, chunk_chars)")
        if not 0.0 <= min_vector_similarity < 1.0:
            raise ValueError("min_vector_similarity must be within [0, 1)")
        self.top_k = top_k
        self.vector_top_k = vector_top_k
        self.text_top_k = text_top_k
        self.vector_only = vector_only
        self.chunk_chars = chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_chunks_per_memory = max_chunks_per_memory
        self.candidate_multiplier = candidate_multiplier
        # 过滤向量路径的弱命中：无关文本之间的余弦并非零（哈希碰撞或真实
        # Embedding 的普遍正值），低于该阈值不参与融合。
        self.min_vector_similarity = min_vector_similarity
        self.snippet_chars = snippet_chars
        self.recall_message_max_chars = recall_message_max_chars
        self.query_max_chars = query_max_chars

@dataclass(frozen=True, slots=True)
class MemorySearchInputs:
    user_message: str
    recent_user_messages: tuple[str, ...] = ()
    summary_goal: str | None = None
    task_title: str | None = None
    task_active_steps: tuple[str, ...] = ()

    def with_task(
        self,
        task_title: str | None,
        task_active_steps: tuple[str, ...],
    ) -> MemorySearchInputs:
        """补上活动 Task 字段（Session 在首次构建时读取一次）。"""

        return replace(self, task_title=task_title, task_active_steps=task_active_steps)

    def render(self, *, max_chars: int = 1_600) -> str:
        """确定性渲染为有界 Query 文本（向量 + 文本 共用）。"""

        sections: list[str] = [self.user_message.strip()]
        recent = [
            message.strip()
            for message in self.recent_user_messages
            if message.strip()
        ]
        if recent:
            sections.append(" | ".join(recent))
        if self.summary_goal:
            sections.append(self.summary_goal.strip())
        if self.task_title:
            task_line = self.task_title.strip()
            if self.task_active_steps:
                task_line += " > " + "; ".join(
                    step.strip() for step in self.task_active_steps if step.strip()
                )
            sections.append(task_line)
        return "\n".join(section for section in sections if section)[:max_chars]
