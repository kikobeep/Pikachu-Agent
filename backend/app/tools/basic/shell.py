"""Shell 命令执行工具（需人工审核）。

危险操作，权限档位为 HUMAN_APPROVAL：模型可以申请调用，
但执行前必须经过 ApprovalGate 的人工确认。
"""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Any

from app.models.types import ToolDefinition, ToolPermission
from app.sandbox import SandboxConfig, SandboxNetworkMode, SandboxSupervisor

from ..base import BaseTool
from ._workspace import resolve_workspace_path, workspace_root_path

MAX_SHELL_TIMEOUT_SECONDS = 120.0


class ShellCommandTool(BaseTool):
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        sandbox_supervisor: SandboxSupervisor | None = None,
    ) -> None:
        self._workspace_root = workspace_root_path(workspace_root)
        self._sandbox_supervisor = sandbox_supervisor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_shell_command",
            description="在本地工作区中执行 Shell 命令，返回退出码、标准输出和标准错误。需要人工审批。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 Shell 命令。",
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "可选的命令执行目录，相对于工作区；默认使用工作区根目录。",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": f"命令执行的最长等待时间，超时后终止，单位为秒，上限为 {MAX_SHELL_TIMEOUT_SECONDS:g} 秒。",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            strict=True,
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("'command' must be a non-empty string")

        cwd = self._workspace_root
        raw_directory = arguments.get("working_directory")
        if raw_directory is not None:
            if not isinstance(raw_directory, str):
                raise ValueError("'working_directory' must be a string")
            cwd = resolve_workspace_path(
                self._workspace_root,
                raw_directory,
                allow_root=True,
            )

        timeout = arguments.get("timeout_seconds", 30.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("'timeout_seconds' must be a positive number")
        timeout = min(float(timeout), MAX_SHELL_TIMEOUT_SECONDS)

        started_at = perf_counter()
        timed_out = False
        if self._sandbox_supervisor is None:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                start_new_session=True,
            )
        else:
            environment = _shell_environment()
            launch = self._sandbox_supervisor.prepare_launch(
                command="/bin/sh",
                args=("-c", command),
                env=environment,
                cwd=str(cwd),
                config=SandboxConfig(network=SandboxNetworkMode.DENIED),
            )
            process = await asyncio.create_subprocess_exec(
                launch.command,
                *launch.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=launch.cwd,
                env=launch.env,
                start_new_session=True,
            )
        try:
            async with asyncio.timeout(timeout):
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            timed_out = True
            _terminate_process(process)
            stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.CancelledError:
            # Run 被 cancel：立即终止整个进程组，避免残留子进程，再重抛取消。
            _terminate_process(process)
            with suppress(Exception):
                await process.communicate()
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        return {
            "command": command,
            "working_directory": cwd.as_posix(),
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "duration_ms": round(duration_ms, 3),
        }


def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """终止整个进程组，避免遗留子进程。"""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _shell_environment() -> dict[str, str]:
    """Shell 只继承执行所需环境，不向命令暴露 Provider 密钥。"""

    safe_keys = ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
    return {key: os.environ[key] for key in safe_keys if os.environ.get(key)}
