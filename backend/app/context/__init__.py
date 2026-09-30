"""context 模块的公共接口。"""

from .blocks import (
    BlockType,
    ConversationBlock,
    MessageBlock,
    SystemBlock,
    ToolRoundBlock,
    partition_messages,
)
from .budget import (
    DEFAULT_SAFETY_MARGIN_TOKENS,
    DEFAULT_TARGET_RATIO,
    DEFAULT_TRIGGER_RATIO,
    ContextBudget,
    ContextBudgetPolicy,
    build_budget_policy,
)
from .capability import (
    CapabilitySource,
    ModelCapabilities,
    ModelCapabilitiesRegistry,
    build_model_capability_registry,
)
from .config import (
    ContextSettings,
    ContextSummaryModelConfig,
)
from .manager import (
    ContextCompactionStage,
    ContextDecision,
    ContextManager,
)
from .summary import (
    SUMMARY_MESSAGE_NAME,
    ConversationSummaryState,
    RollingConversationSummary,
    SummaryGenerationResult,
)
from .summary_store import ConversationSummaryStore
from .tokens import (
    DEFAULT_ENCODING,
    DEFAULT_FAMILY_FACTORS,
    TokenEstimator,
    default_token_estimator,
    model_family,
)
from . import reducers

__all__ = [
    'BlockType',
    'CapabilitySource',
    'ContextBudget',
    'ContextBudgetPolicy',
    'ContextCompactionStage',
    'ContextDecision',
    'ContextManager',
    'ContextSettings',
    'ContextSummaryModelConfig',
    'ConversationBlock',
    'ConversationSummaryState',
    'ConversationSummaryStore',
    'DEFAULT_ENCODING',
    'DEFAULT_FAMILY_FACTORS',
    'DEFAULT_SAFETY_MARGIN_TOKENS',
    'DEFAULT_TARGET_RATIO',
    'DEFAULT_TRIGGER_RATIO',
    'MessageBlock',
    'ModelCapabilities',
    'ModelCapabilitiesRegistry',
    'RollingConversationSummary',
    'SUMMARY_MESSAGE_NAME',
    'SummaryGenerationResult',
    'SystemBlock',
    'TokenEstimator',
    'ToolRoundBlock',
    'build_budget_policy',
    'build_model_capability_registry',
    'default_token_estimator',
    'model_family',
    'partition_messages',
    'reducers',
]
