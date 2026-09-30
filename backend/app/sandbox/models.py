from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxFilesystemMode(StrEnum):
    """沙箱对 workspace 的访问级别。"""

    NONE = "none"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    HOST = "host"


class SandboxNetworkMode(StrEnum):
    """当前原生后端支持的网络策略。"""

    DENIED = "denied"
    UNRESTRICTED = "unrestricted"


class SandboxConfig(BaseModel):
    """用户可持久化的沙箱配置；默认限制文件系统但保留 MCP 联网能力。"""

    model_config = ConfigDict(extra="forbid")

    filesystem: SandboxFilesystemMode = SandboxFilesystemMode.WORKSPACE_WRITE
    network: SandboxNetworkMode = SandboxNetworkMode.UNRESTRICTED
    readable_roots: tuple[str, ...] = ()
    writable_roots: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()

    @field_validator("readable_roots", "writable_roots", "allowed_domains")
    @classmethod
    def reject_empty_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("sandbox list values cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("sandbox list values cannot contain duplicates")
        return normalized


class SandboxPolicy(BaseModel):
    """解析为绝对路径后交给平台后端强制执行的不可变策略。"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    filesystem: SandboxFilesystemMode
    network: SandboxNetworkMode
    working_directory: Path
    readable_roots: tuple[Path, ...]
    writable_roots: tuple[Path, ...]
    denied_read_paths: tuple[Path, ...] = ()
    denied_write_paths: tuple[Path, ...] = ()
    allowed_domains: tuple[str, ...] = ()


class SandboxLaunchSpec(BaseModel):
    """平台后端编译后的进程启动参数。"""

    model_config = ConfigDict(frozen=True)

    command: str
    args: tuple[str, ...]
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    backend: str
    sandboxed: bool


__all__ = [
    "SandboxConfig",
    "SandboxFilesystemMode",
    "SandboxLaunchSpec",
    "SandboxNetworkMode",
    "SandboxPolicy",
]
