"""统一执行工具调用，检查权限并记录结果。"""

import asyncio
import json
from time import perf_counter

from app.types import AgentMode
from .base import ToolCall, ToolResult, ToolPermission
from .register import ToolRegistry

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._registry = registry
        self._timeout_seconds = timeout_seconds

    async def execute(
        self,
        call: ToolCall,
        *,
        mode: AgentMode = AgentMode.DEFAULT,
    ) -> ToolResult:
        started_at = perf_counter()
        output = None
        error = None

        try:
            tool = self._registry.get(call.name)
        except KeyError:
            error = f"Tool {call.name!r} is not registered"
        else:
            if not self._registry.is_allowed_for_mode(call.name, mode):
                error = f"Tool {call.name!r} is not allowed in {mode.value} mode"

            elif tool.definition.permission is ToolPermission.FORBIDDEN:
                error = f"Tool {call.name!r} is forbidden"

            elif tool.definition.permission is ToolPermission.HUMAN_APPROVAL:
                error = "This tool requires approval; approval is not implemented"

            else:
                try:
                    arguments = call.arguments
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    async with asyncio.timeout(self._timeout_seconds):
                        output = await tool.execute(arguments)
                except TimeoutError:
                    error = (
                        f"Tool timed out after "
                        f"{self._timeout_seconds:g} seconds"
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            status=error is None,
            output=output,
            error=error,
            duration_ms=(perf_counter() - started_at) * 1000,
        )
        


def main() -> None:
    import argparse
    from uuid import uuid4

    from .basic import basic_tool_registry

    parser = argparse.ArgumentParser(description="通过 ToolExecutor 执行 web_search")
    parser.add_argument("query", help="搜索词")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--topic", choices=["general", "news", "finance"], default="general")
    parser.add_argument("--mode", choices=[mode.value for mode in AgentMode], default="default")
    parser.add_argument("--timeout", type=float, default=30.0, help="执行器超时秒数")
    args = parser.parse_args()
    arguments = {"query": args.query, "topic": args.topic}
    if args.max_results is not None:
        arguments["max_results"] = args.max_results
    try:
        registry = basic_tool_registry(tool_names=("web_search",))
        executor = ToolExecutor(registry, timeout_seconds=args.timeout)
        call = ToolCall(id=str(uuid4()), name="web_search", arguments=arguments)
        result = asyncio.run(executor.execute(call, mode=AgentMode(args.mode)))
    except ValueError as exc:
        parser.exit(1, f"初始化失败：{exc}\n")
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if not result.status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
