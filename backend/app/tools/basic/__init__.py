"""基础的内置工具。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.sandbox import SandboxSupervisor

from ..register import ToolRegistry
from ..search import SearchSettings


def basic_tool_registry(
    workspace_root: str | Path | None = None,
    *,
    search_settings: SearchSettings | None = None,
    tool_names: tuple[str, ...] | None = None,
    sandbox_supervisor: SandboxSupervisor | None = None,
) -> ToolRegistry:
    """创建注册表；tool_names 可选择工具，省略时注册全部内置工具。"""

    available = {
        "get_current_time": "CurrentTimeTool",
        "list_files": "ListFilesTool",
        "read_file": "ReadFileTool",
        "write_file": "WriteFileTool",
        "run_shell_command": "ShellCommandTool",
        "http_request": "HttpRequestTool",
        "web_search": "WebSearchTool",
    }
    selected = tuple(available) if tool_names is None else tool_names
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError(f"Unknown basic tools: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("tool_names must not contain duplicates")

    registry = ToolRegistry()
    for name in selected:
        tool_class = __getattr__(available[name])
        if name == "web_search":
            tool = tool_class(settings=search_settings)
        elif name == "run_shell_command":
            tool = tool_class(workspace_root, sandbox_supervisor=sandbox_supervisor)
        elif name in {"list_files", "read_file", "write_file"}:
            tool = tool_class(workspace_root)
        else:
            tool = tool_class()
        registry.register(tool)
    return registry


__all__ = [
    "CurrentTimeTool",
    "HttpRequestTool",
    "ListFilesTool",
    "ReadFileTool",
    "ShellCommandTool",
    "WebSearchTool",
    "WriteFileTool",
    "basic_tool_registry",
]


_TOOL_MODULES = {
    "CurrentTimeTool": "current_time",
    "HttpRequestTool": "http_request",
    "ListFilesTool": "list_files",
    "ReadFileTool": "read_file",
    "ShellCommandTool": "shell",
    "WebSearchTool": "web_search",
    "WriteFileTool": "write_file",
}


def __getattr__(name: str):
    """按需导入工具，让独立搜索不依赖其他工具的运行环境。"""
    if name not in _TOOL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_TOOL_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
