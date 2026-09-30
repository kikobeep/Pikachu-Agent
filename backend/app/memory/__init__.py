"""memory 模块的公共接口。"""

from .archive import MemoryArchive
from .archive_config import (
    ArchiveAction,
    MemoryArchiveConfig,
    MemoryArchiveDecision,
    MemoryArchiveInput,
    MemoryArchiveResponse,
)
from .core import CoreMemory, CoreMemoryRecord
from .embedding import (
    EmbeddingAdapter,
    HashEmbeddingAdapter,
    MemoryEmbeddingSettings,
    RemoteEmbeddingAdapter,
    build_embedding_adapter,
)
from .embedding_registry import EmbeddingAdapterRegistry
from .manager import MemoryManager
from .reflection_config import (
    MemoryReflectionConfig,
    MemoryReflectionInput,
    MemoryReflectionReponse,
    ReflectionAction,
    ReflectionDecision,
)
from .reflection_gate import (
    ReflectionGateDecision,
    ReflectionGateReason,
    decide_reflection_gate,
)
from .regular import MemoryStatus, RegularMemoryIndex, RegularMemoryRecord
from .search import MemorySearchService, recent_user_message_texts
from .search_config import (
    DEFAULT_SEARCH_DATABASE_NAME,
    MemorySearchConfig,
    MemorySearchCandidate,
    MemorySearchInputs,
    MemorySearchResult,
    SearchMode,
)
from .store import DEFAULT_MEMORY_DIR, MemoryStore
from .tools import (
    DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES,
    MEMORY_SEARCH_TOOL_NAME,
    MemoryArchiveTool,
    MemoryCreateTool,
    MemoryListTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryUpdateTool,
    CoreMemoryRemoveTool,
    CoreMemoryUpdateTool,
    register_memory_tools,
    register_memory_write_tools,
)

__all__ = [
    'ArchiveAction',
    'CoreMemory',
    'CoreMemoryRecord',
    'CoreMemoryRemoveTool',
    'CoreMemoryUpdateTool',
    'DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES',
    'DEFAULT_MEMORY_DIR',
    'DEFAULT_SEARCH_DATABASE_NAME',
    'EmbeddingAdapter',
    'EmbeddingAdapterRegistry',
    'HashEmbeddingAdapter',
    'MEMORY_SEARCH_TOOL_NAME',
    'MemoryArchive',
    'MemoryArchiveConfig',
    'MemoryArchiveDecision',
    'MemoryArchiveInput',
    'MemoryArchiveResponse',
    'MemoryArchiveTool',
    'MemoryCreateTool',
    'MemoryEmbeddingSettings',
    'MemoryListTool',
    'MemoryManager',
    'MemoryReadTool',
    'MemoryReflectionConfig',
    'MemoryReflectionInput',
    'MemoryReflectionReponse',
    'MemorySearchCandidate',
    'MemorySearchConfig',
    'MemorySearchInputs',
    'MemorySearchResult',
    'MemorySearchService',
    'MemorySearchTool',
    'MemoryStatus',
    'MemoryStore',
    'MemoryUpdateTool',
    'ReflectionAction',
    'ReflectionDecision',
    'ReflectionGateDecision',
    'ReflectionGateReason',
    'RegularMemoryIndex',
    'RegularMemoryRecord',
    'RemoteEmbeddingAdapter',
    'SearchMode',
    'build_embedding_adapter',
    'decide_reflection_gate',
    'recent_user_message_texts',
    'register_memory_tools',
    'register_memory_write_tools',
]
