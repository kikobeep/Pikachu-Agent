"""按配置创建、缓存 Embedding Adapter。"""
from __future__ import annotations

from collections.abc import Callable

from .embedding import (
    EmbeddingAdapter,
    HashEmbeddingAdapter,
    MemoryEmbeddingSettings,
    RemoteEmbeddingAdapter,
)

EmbeddingAdapterFactory = Callable[[MemoryEmbeddingSettings], EmbeddingAdapter]


def _remote_adapter(settings: MemoryEmbeddingSettings) -> EmbeddingAdapter:
    if not settings.base_url or not settings.model:
        raise ValueError("remote embedding requires base_url and model")
    return RemoteEmbeddingAdapter(
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        model=settings.model,
        dimensions=settings.dimensions,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        batch_size=settings.batch_size,
    )


def _qwen3_adapter(settings: MemoryEmbeddingSettings) -> EmbeddingAdapter:
    # 百炼官方 Qwen3 Embedding 系列的 API 模型名是 text-embedding-v4。
    config = settings.model_copy(update={
        "base_url": settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": settings.model or "text-embedding-v4",
        "batch_size": min(settings.batch_size, 10),
    })
    return _remote_adapter(config)


def _bge_adapter(settings: MemoryEmbeddingSettings) -> EmbeddingAdapter:
    if not settings.base_url:
        raise ValueError("bge-zh-small requires the URL of a compatible embedding service")
    return _remote_adapter(settings.model_copy(update={
        "model": settings.model or "BAAI/bge-small-zh-v1.5",
    }))


def _hash_adapter(settings: MemoryEmbeddingSettings) -> EmbeddingAdapter:
    return HashEmbeddingAdapter(
        dimensions=settings.dimensions or 1024,
        model_name=settings.model or "hash-embedding",
    )


class EmbeddingAdapterRegistry:
    """与普通模型 Registry 一样按名称创建并复用适配器。

    settings 配置默认 provider；其他 provider 可通过 register(settings=...)
    指定独立配置。实例创建后不允许替换其工厂，以免遗失待关闭的客户端。
    """

    def __init__(self, settings: MemoryEmbeddingSettings | None = None) -> None:
        self.settings = settings or MemoryEmbeddingSettings()
        self._factories: dict[str, EmbeddingAdapterFactory] = {
            "qwen3-embedding": _qwen3_adapter,
            "bge-zh-small": _bge_adapter,
            "hash": _hash_adapter,
        }
        self._configs: dict[str, MemoryEmbeddingSettings] = {}
        self._instances: dict[str, EmbeddingAdapter] = {}

    def register(
        self,
        provider: str,
        factory: EmbeddingAdapterFactory,
        *,
        settings: MemoryEmbeddingSettings | None = None,
        replace: bool = False,
    ) -> None:
        name = provider.strip().lower()
        if not name:
            raise ValueError("embedding provider cannot be empty")
        if name in self._instances:
            raise ValueError("close the registry before replacing an active adapter")
        if name in self._factories and not replace:
            raise ValueError(f"Embedding provider already registered: {name}")
        self._factories[name] = factory
        if settings is not None:
            self._configs[name] = settings
        else:
            self._configs.pop(name, None)

    def get(self, provider: str | None = None) -> EmbeddingAdapter | None:
        name = (provider if provider is not None else self.settings.provider).strip().lower()
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"Unsupported embedding provider: {name}")
        if name in self._instances:
            return self._instances[name]
        config = self._configs.get(name, self.settings)
        if not config.enabled:
            return None
        # 保留未配置远程服务时返回 None、降级为文本检索的行为。
        if name in {"qwen3-embedding", "bge-zh-small"} and (
            config.api_key is None or not config.api_key.get_secret_value().strip()
        ):
            return None
        adapter = factory(config)
        self._instances[name] = adapter
        return adapter

    async def close(self) -> None:
        for adapter in tuple(self._instances.values()):
            await adapter.close()
        self._instances.clear()
