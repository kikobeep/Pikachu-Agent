"""最小工具调用 Agent 运行时使用的错误。"""


class AgentRuntimeError(RuntimeError):
    """Agent 运行时错误的基类。"""


class ModelInvocationError(AgentRuntimeError):
    """模型注册表或适配器无法完成请求时抛出。"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"model invocation failed: {detail}")


class ContextPreparationError(AgentRuntimeError):
    """模型请求上下文无法完成准备时抛出。"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"context preparation failed: {detail}")


class ContextWindowExceededError(ContextPreparationError):
    """估算输入超过可用输入预算时抛出。"""

    def __init__(self, estimated_tokens: int, input_budget: int) -> None:
        super().__init__(
            f"estimated input tokens ({estimated_tokens}) exceed "
            f"input budget ({input_budget})"
        )


class MaxStepsExceededError(AgentRuntimeError):
    """运行时达到配置的最大步数时抛出。"""

    def __init__(self, max_steps: int) -> None:
        super().__init__(f"maximum step limit ({max_steps}) reached")


class RunBudgetExceededError(AgentRuntimeError):
    """Main Agent 已达到 Run 硬预算，不能继续请求模型。"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"main agent run budget exceeded: {detail}")


class RepeatedToolCallError(AgentRuntimeError):
    """模型连续三次重复同一工具调用时抛出。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"tool {tool_name!r} was called with identical arguments "
            "3 consecutive times"
        )
