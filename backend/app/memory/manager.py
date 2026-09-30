
from __future__ import annotations

import logging

from app.model.config import Message, MessageRole
from app.memory.embedding import EmbeddingAdapter
from app.memory.prompt import CORE_MEMORY_HEADER, MEMORY_POLICY_PROMPT
from app.memory.search import MemorySearchService
from app.memory.search_config import (
    DEFAULT_SEARCH_DATABASE_NAME,
    MemorySearchConfig,
    MemorySearchInputs,
    MemorySearchResult,
)
from app.memory.search_index import MemorySearchIndex

logger = logging.getLogger(__name__)

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from .archive import MemoryArchive
from .core import CoreMemory
from .regular import MemoryStatus, RegularMemoryIndex, RegularMemoryRecord
from .store import DEFAULT_MEMORY_DIR, MemoryStore

DEFAULT_MAX_CORE_TOKENS = 2000


class MemoryManager:
    def __init__(
        self,
        memory_dir: str | Path = DEFAULT_MEMORY_DIR,
        max_core_tokens: int = DEFAULT_MAX_CORE_TOKENS,
        max_active: int = 100,
        embedding: EmbeddingAdapter | None = None,
        search_configs: MemorySearchConfig | None = None,
        min_vector_similarity: float | None = None,
        hybrid_search_enabled: bool = True,
    ):
        # 核心记忆 & 普通记忆
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.core = CoreMemory(memory_dir=self.memory_dir, max_tokens=max_core_tokens)
        self.max_active = max_active
        self.store = MemoryStore(self.memory_dir, max_active=max_active)
        self.regular = RegularMemoryIndex(memory_dir=self.memory_dir)
        # self.index = RegularMemoryIndex(self.memory_dir)
        self._archive = MemoryArchive(max_active=max_active)

        # 检索 & 召回
        if search_configs is None and min_vector_similarity is not None:
            search_configs = MemorySearchConfig(
                min_vector_similarity=min_vector_similarity
            )
        self.search_configs = search_configs or MemorySearchConfig()
        self._hybrid_search_enabled = hybrid_search_enabled
        self._search_index = MemorySearchIndex(
            self.memory_dir / DEFAULT_SEARCH_DATABASE_NAME,
            embedding=embedding,
            configs=self.search_configs,
        )
        self._search_ok = False

        self._search_service = MemorySearchService(store = self.store, search_index = self._search_index)
        
        self._lock = asyncio.Lock()

    @property
    def hybrid_search_enabled(self) -> bool:
        """Hybrid 自动召回是否可用（索引初始化成功且未被关闭）。"""

        return self._hybrid_search_enabled and self._search_ok
    
    async def initialize(self) -> None:
        async with self._lock:
            await self.store.initialize()
            await self.core.initialize()
            await self._rebuild_regular_memory_index()
        if self._hybrid_search_enabled:
            try:
                await self._search_index.initialize()
                # 用最新的记忆更新 search index
                await self._search_index.reconcile(
                    await self.store.list_active(),
                    generate_embeddings=False,
                )
                self._search_ok = True
                # 为数据库中缺失向量、或者使用了旧模型的文本片段，重新生成并保存向量
                await self._search_index.generate_embeddings()
            except Exception as exc:
                self._search_ok = False
                logger.warning(
                    "memory search index unavailable, fallback to index cues: %s",
                    exc,
                )


    async def reflection_context(self) -> tuple[str, str]:
        """返回 Post-Run Reflection 使用的 Core 正文与当前 Index。"""
        async with self._lock:
            core_memory = await self.core.load() or ""
            regular_memory = await self.regular.load() or ""
            return core_memory, regular_memory
        


    async def memory_context(self, search_result: MemorySearchResult | None = None) -> tuple[Message, ...]:
        async with self._lock:
            messages: list[Message] = []
            # 加载核心记忆
            core_text = await self.core.load()
            if core_text.strip():
                core_content = core_text.strip()
                if not core_content.startswith(CORE_MEMORY_HEADER):
                    core_content = f"{CORE_MEMORY_HEADER}\n\n{core_content}"
                messages.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=core_content,
                    )
                )
            # 如果普通记忆的语义+向量召回结果为空，则使用
            if search_result is None:
                regular_text = await self.regular.load()
                if regular_text is not None:
                    messages.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            name="memory_index",
                            content=regular_text,
                        )
                    )
            else:
                search_messages = search_result.render_message(
                    max_chars=self.search_configs.recall_message_max_chars
                )
                messages.append(search_messages)


            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    name="memory_policy",
                    content=MEMORY_POLICY_PROMPT,
                )
            )
            return tuple(messages)



    async def update_regular_memory(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        title: str,
        summary: str,
        content: str,
        reason: str,
    ) -> RegularMemoryRecord:
        # 同一管理器内串行更新，避免并发写入
        async with self._lock:
            record = await self.store.update(
                memory_id,
                title=title,
                summary=summary,
                content=content,
                reason=reason,
                expected_revision=expected_revision,
            )
            # 同步更新index.md
            await self._rebuild_regular_memory_index()
            await self._reconcile_search_index()
            return record

    async def _rebuild_regular_memory_index(self) -> None:
        await self.regular.rebuild(await self.store.list_active())

    async def _reconcile_search_index(self) -> None:
        """同步 active memories 到 BM25/向量搜索索引。"""
        if not self._search_ok:
            return
        await self._search_index.reconcile(
            await self.store.list_active(),
            generate_embeddings=False,
        )

    async def refresh_search_embeddings(self) -> None:
        """为索引中缺失或过期的 memory chunk 回填向量。"""
        if not self._search_ok:
            return
        await self._search_index.generate_embeddings()

    async def active_count(self) -> int:
        """返回当前 active Memory 数量。"""

        async with self._lock:
            return await self.store.count_active()

    async def get_candidates(
        self,
        *,
        limit: int = 5,
    ) -> tuple[RegularMemoryRecord, ...]:
        """返回最可能值得维护的候选（最终决策交给模型）。"""

        async with self._lock:
            active = await self.store.list_active()
            return self._archive.select_candidates(active, limit=limit)

    async def archive_if_unchanged(
        self,
        memory_id: str,
        *,
        expected_record: RegularMemoryRecord,
        reason: str,
    ) -> RegularMemoryRecord:
        """候选快照仍为最新时归档，拒绝维护模型基于陈旧内容执行。"""

        async with self._lock:
            record = await self.store.archive(
                memory_id,
                expected_record=expected_record,
                reason=reason,
            )
            await self._rebuild_regular_memory_index()
            await self._reconcile_search_index()
            return record

    async def create_if_capacity(
        self,
        *,
        title: str,
        summary: str,
        content: str,
    ):
        async with self._lock:
            # 写锁保护范围内再次检查
            if await self.store.count_active() >= self.max_active:
                return None
            record = await self.store.create(
                title=title, summary=summary, content=content
            )
            await self._rebuild_regular_memory_index()
            await self._reconcile_search_index()
            return record

    async def is_over_capacity(self) -> bool:
        """active 数量是否超过上限。"""

        async with self._lock:
            return self._archive.exceeds_capacity(await self.store.count_active())

    async def list(self) -> tuple[RegularMemoryRecord, ...]:
        """列出当前 active 记忆（id / title / summary）。"""

        async with self._lock:
            return await self.store.list_active()
    
    async def list_archived(self) -> tuple[RegularMemoryRecord, ...]:
        """列出已归档记忆，仅供管理与观察界面读取。"""

        async with self._lock:
            return await self.store.list_archived()

    async def read(self, memory_id: str) -> RegularMemoryRecord | None:
        """读取一条记忆并记录访问（access_count / last_accessed_at）。"""

        record = await self.store.load(memory_id)
        if record is None or record.status is not MemoryStatus.ACTIVE:
            return record
        updated = RegularMemoryRecord(**{
            **record.model_dump(),
            "access_count": record.access_count + 1,
            "last_accessed_at": datetime.now(UTC),
        })
        await self.store.write(updated)
        return updated

    async def update_if_revision(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        title: str,
        summary: str,
        content: str,
        reason: str,
    ) -> RegularMemoryRecord:
        return await self.update_regular_memory(
            memory_id,
            expected_revision=expected_revision,
            title=title,
            summary=summary,
            content=content,
            reason=reason,
        )

    async def archive(self, memory_id: str, *, reason: str) -> RegularMemoryRecord:
        async with self._lock:
            record = await self.store.archive(memory_id, reason=reason)
            await self._rebuild_regular_memory_index()
            await self._reconcile_search_index()
            return record

    async def upsert_core(
        self,
        *,
        key: str,
        value: str,
        reason: str,
        source_statement: str,
    ):
        return await self.core.insert(
            key=key,
            value=value,
            reason=reason,
            source_statement=source_statement,
        )

    async def remove_core(self, key: str):
        return await self.core.remove(key)

    async def search(
        self,
        inputs: MemorySearchInputs,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        """执行一次自动召回（Harness 每 Run 只调用一次）。"""

        return await self._search_service.search(inputs, limit=limit)
