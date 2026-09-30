"""模型适配层抛出的错误。"""


class ModelAdapterError(RuntimeError):
    """模型提供商配置和调用失败的基础错误。"""


class ProviderNotConfiguredError(ModelAdapterError):
    def __init__(self, provider: str, environment_variable: str) -> None:
        super().__init__(
            f"Provider '{provider}' is not configured. "
            f"Set {environment_variable} in the environment."
        )


class UnsupportedProviderError(ModelAdapterError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"No model adapter is registered for provider '{provider}'.")


class UnsupportedMessageError(ModelAdapterError):
    """消息无法转换为模型提供商 API 格式时抛出。"""
