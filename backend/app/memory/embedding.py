
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".config"


class MemoryEmbeddingSettings(BaseSettings):
    """Embedding 服务的独立运行配置（与主模型 Provider 解耦）。"""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MEMORY_EMBEDDING_",
        extra="ignore",
    )

    enabled: bool = False
    provider: str = "qwen3-embedding"
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    dimensions: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=1, ge=0)
    batch_size: int = Field(default=16, ge=1)
    min_similarity: float | None = Field(default=None, ge=0.0, lt=1.0)

    @field_validator("base_url", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("embedding base_url and model must be strings")
        return value.strip() or None

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("embedding provider cannot be empty")
        return value

    def adapter_configured(self) -> bool:
        """Hash 无需密钥；内置远程服务使用默认模型或显式模型。"""
        return self.enabled and (
            self.provider == "hash"
            or (self.api_key is not None and bool(self.api_key.get_secret_value().strip()))
        )


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """最小 Embedding 接口：模型名 + 两个向量化入口。"""

    @property
    def model_name(self) -> str:
        """用于索引对账的模型标识；换模型触发全量重算。"""

    @property
    def dimensions(self) -> int | None:
        """已知向量维度；未知返回 None（由首批结果确定）。"""

    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """把一批记忆 Chunk 文本向量化（索引写入路径）。"""

    async def embed_query(self, text: str) -> tuple[float, ...]:
        """把检索 Query 向量化（查询路径）。"""

    async def close(self) -> None:
        """释放底层客户端资源。"""


class RemoteEmbeddingAdapter:
    """通过 OpenAI 兼容 /embeddings 端点提供向量服务。"""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        batch_size: int = 16,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI

        if not model:
            raise ValueError("embedding model is required")
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if base_url and _is_local_base_url(base_url):
            client_kwargs["http_client"] = httpx.AsyncClient(trust_env=False)
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed(list(batch)))
        return tuple(vectors)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = await self._embed([text])
        return vectors[0]

    async def _embed(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]:
        request: dict[str, object] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: item.index)
        return tuple(
            tuple(float(value) for value in item.embedding) for item in ordered
        )

    async def close(self) -> None:
        await self._client.close()


class HashEmbeddingAdapter:
    """确定性离线向量：哈希 n-gram 袋 + L2 归一化。

    不调用任何外部服务；语义上等价于"共享词元越多越相似"，足以驱动
    离线召回 / 排名 / 降级测试。相同文本永远得到相同向量。
    """

    def __init__(
        self,
        *,
        dimensions: int = 1024,
        model_name: str = "hash-embedding",
    ) -> None:
        if dimensions <= 0:
            raise ValueError("hash embedding dimensions must be positive")
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    async def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self._vector(text)

    async def close(self) -> None:
        return None

    def _vector(self, text: str) -> tuple[float, ...]:
        counts: dict[int, float] = {}
        for token in _semantic_tokens(text):
            bucket = _hash_bucket(token, self._dimensions)
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        vector = [0.0] * self._dimensions
        for bucket, weight in counts.items():
            vector[bucket] = weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return tuple(vector)


_TOKEN_SEPARATOR = re.compile(
    r"[\s,.;:!?，。；：！？、()\[\]{}\"'`|/\\<>@#$%^&*+=~\-_""]+"
)

# 视为"本地端点"的主机名：这些地址上的 Embedding 服务不需要也不应该经过代理。
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})


def _is_local_base_url(base_url: str) -> bool:
    """判断 base_url 是否指向本机（用于决定是否绕过系统代理）。"""

    from urllib.parse import urlsplit

    host = (urlsplit(base_url).hostname or "").strip().lower()
    return host in _LOCAL_HOSTS


def _semantic_tokens(text: str) -> list[str]:
    """提取词元 + CJK bigram，让中文改写查询仍能命中主题相近的记忆。"""

    tokens: list[str] = []
    for raw in _TOKEN_SEPARATOR.split(text.casefold()):
        if not raw:
            continue
        if raw.isascii():
            tokens.append(raw)
            continue
        tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
        if len(raw) == 1:
            tokens.append(raw)
    return tokens


def _hash_bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def build_embedding_adapter(
    settings: MemoryEmbeddingSettings,
) -> EmbeddingAdapter | None:
    """兼容现有调用入口；返回的 Adapter 由调用方负责 close。"""
    from .embedding_registry import EmbeddingAdapterRegistry

    return EmbeddingAdapterRegistry(settings).get()


__all__ = [
    "EmbeddingAdapter",
    "HashEmbeddingAdapter",
    "MemoryEmbeddingSettings",
    "RemoteEmbeddingAdapter",
    "build_embedding_adapter",
]
