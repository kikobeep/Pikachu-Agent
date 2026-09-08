"""统一检查权限、等待审批并执行工具。"""

import asyncio
import json
import math
from copy import deepcopy
from time import perf_counter

from app.types import AgentMode
from .approval import ApprovalGate, ConsoleApprovalGate
from .base import ToolCall, ToolResult, ToolExecutionContext
from .permission import PermissionHook
from .register import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 30.0,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._approval_gate = approval_gate
        self._permission_hook = PermissionHook()

    async def execute(
        self,
        call: ToolCall,
        *,
        mode: AgentMode = AgentMode.DEFAULT,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ToolResult:
        started_at = perf_counter()
        # 隔离调用方可变对象，后续审批、执行和结果都对应同一次调用。
        call = call.model_copy(deep=True)
        output = None
        error = None
        try:
            try:
                tool = self._registry.get(call.name)
            except KeyError:
                raise ValueError(f"Tool {call.name!r} is not registered") from None
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            # 固定为 JSON 数据，拒绝 NaN 或不能序列化的对象。
            arguments_json = json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
            arguments = json.loads(arguments_json)
            context = ToolExecutionContext(
                tool_call=call.model_copy(deep=True),
                tool_definition=tool.definition.model_copy(deep=True),
                arguments=deepcopy(arguments),
                mode=mode,
                run_id=run_id,
                conversation_id=conversation_id,
            )
            await self._authorize(context)
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    output = await tool.execute(arguments)
            except TimeoutError:
                raise TimeoutError(f"Tool timed out after {self._timeout_seconds:g} seconds") from None
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


    async def _authorize(self, context: ToolExecutionContext) -> None:
        """检查模式、权限和审批；正常返回表示允许，拒绝时抛出异常。"""
        # 调用 PermissionHook 判断权限。需要审批时调用 ApprovalGate
        call_id = context.tool_call.id
        tool_name = context.tool_call.name
        mode = context.mode
        if mode is None:
            raise PermissionError("Missing execution mode")
        if not self._registry.is_allowed_for_mode(tool_name, mode):
            raise PermissionError(f"Tool {tool_name!r} is not allowed in {mode.value} mode")
        if context.arguments is None:
            raise PermissionError("Missing parsed tool arguments")
        arguments_json = json.dumps(
            context.arguments, ensure_ascii=False, sort_keys=True, allow_nan=False,
        )
        decision = await self._permission_hook.before_execute(context)
        if decision is None:
            return
        if decision.denied_reason is not None:
            raise PermissionError(decision.denied_reason)
        request = decision.approval_request
        if request is None:
            return
        if self._approval_gate is None:
            raise PermissionError("This tool requires approval, but no approval gate is configured")
       
        approved = await self._approval_gate.request(request)
        if approved is not True:
            raise PermissionError("Tool execution was not approved")
        approved_json = json.dumps(
            request.arguments, ensure_ascii=False, sort_keys=True, allow_nan=False,
        )
        if (request.tool_call_id != call_id or request.tool_name != tool_name
                or approved_json != arguments_json):
            raise PermissionError("Approval request changed; refusing to execute")


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
        executor = ToolExecutor(registry, timeout_seconds=args.timeout, approval_gate=ConsoleApprovalGate())
        call = ToolCall(id=str(uuid4()), name="web_search", arguments=arguments)
        result = asyncio.run(executor.execute(call, mode=AgentMode(args.mode)))
    except ValueError as exc:
        parser.exit(1, f"初始化失败：{exc}\n")
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if not result.status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
