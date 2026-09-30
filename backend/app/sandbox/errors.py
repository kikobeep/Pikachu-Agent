"""沙箱领域错误。"""


class SandboxError(RuntimeError):
    """可识别的沙箱基础错误。"""


class SandboxPolicyError(SandboxError):
    """请求的沙箱策略无效或无法安全解析。"""


class SandboxUnavailableError(SandboxError):
    """当前平台无法落实请求的隔离策略。"""


__all__ = ["SandboxError", "SandboxPolicyError", "SandboxUnavailableError"]
