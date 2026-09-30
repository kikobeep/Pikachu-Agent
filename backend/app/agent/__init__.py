"""agent 模块的公共接口。"""

from app.model.config import (
    AgentMode,
    Message,
    MessageRole,
    ToolCall,
)
from .budget import (
    RunBudget,
    RunBudgetConfig,
    RunBudgetDecision,
    RunBudgetReason,
    RunBudgetStatus,
    chargeable_tokens,
)
from .emitter import EventEmitter
from .error import (
    AgentRuntimeError,
    ContextPreparationError,
    ContextWindowExceededError,
    MaxStepsExceededError,
    ModelInvocationError,
    RepeatedToolCallError,
    RunBudgetExceededError,
)
from .events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
    MultiEventHandlers,
)
from .loop import AgentLoop
from .result import (
    AgentError,
    AgentResult,
    AgentStopReason,
    ToolCallRecord,
    ToolRound,
)
from .runtime import AgentRuntime
from .spec import load_agent_prompt, resolve_agent_md_path
from .tool_hooks import (
    AgentEventEmitter,
    AgentEventHook,
)

__all__ = [
    'AgentError',
    'AgentEvent',
    'AgentEventEmitter',
    'AgentEventHandler',
    'AgentEventHook',
    'AgentEventType',
    'AgentLoop',
    'AgentMode',
    'AgentResult',
    'AgentRuntime',
    'AgentRuntimeError',
    'AgentStopReason',
    'ContextPreparationError',
    'ContextWindowExceededError',
    'EventEmitter',
    'MaxStepsExceededError',
    'Message',
    'MessageRole',
    'ModelInvocationError',
    'MultiEventHandlers',
    'RepeatedToolCallError',
    'RunBudget',
    'RunBudgetConfig',
    'RunBudgetDecision',
    'RunBudgetExceededError',
    'RunBudgetReason',
    'RunBudgetStatus',
    'ToolCall',
    'ToolCallRecord',
    'ToolRound',
    'chargeable_tokens',
    'load_agent_prompt',
    'resolve_agent_md_path',
]
