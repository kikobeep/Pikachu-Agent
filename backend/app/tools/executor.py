"""统一检查权限、等待审批并执行工具。"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any
from datetime import UTC, datetime

from app.model.config import AgentMode, ToolCall
from app.model.adapter import _parse_arguments
from app.tools.config import BaseTool, ToolResult
from .config import ToolExecutionContext
from .hooks import (
    ToolHook,
    ToolHookDecision,
    ToolHookRunner,
)
from app.tools.output import ToolOutputRecorder
from app.tools.permissions.models import ApprovalDecision
from app.tools.permissions.policy import PermissionPolicyEngine
from app.tools.permissions.rule_factory import build_safe_rule
from app.tools.permissions.store import PermissionRuleStore

from .approval import ApprovalGate, ConsoleApprovalGate,DenyAllGate
from .permission import PermissionHook
from .register import ToolRegistry

MAX_TOOL_OUTPUT_CHARS = 20_000

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 30.0,
        max_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
        approval_gate: ApprovalGate | None = None,
        # logger: ToolExecutionLogger | None = None,
        hooks: Sequence[ToolHook] = (),
        policy_engine: PermissionPolicyEngine | None = None,
        rule_store: PermissionRuleStore | None = None,
        rule_factory: Any = build_safe_rule,
        output_recorder: ToolOutputRecorder | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._approval_gate = approval_gate

        self.rule_store = rule_store # 保存已记住的权限规则，不用再反复询问用户

        resolved_store = rule_store or (
            policy_engine.store if policy_engine is not None else None
        )
        resolved_policy = policy_engine or (    # 读取已保存的规则，根据当前工具的name及参数，对应 ALLOW、DENY 还是 ASK
            PermissionPolicyEngine(resolved_store)
            if resolved_store is not None
            else None
        )

        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars
        # self.logger = logger or InMemoryExecutionLogger()
        self._permission_hook = PermissionHook( # 把权限相关的流程串起来
            approval_gate or DenyAllGate(),
            policy=resolved_policy,
            rule_store=resolved_store,
            rule_factory=rule_factory,
        )
        self._hooks = (
            self._permission_hook, #检查权限、处理审批
            # ObservabilityHook(self.logger), # 记录工具执行日志
            *hooks, # 扩展的处理流程，比如发送工具开始／结束事件
        )
        self._output_recorder = output_recorder
    
    
    async def clear_run_rules(self, run_id: str) -> int:
        """清理一次 Agent Run 创建的临时审批规则。"""
        return await self._permission_hook.clear_run_rules(run_id)
    
    async def execute(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
        hooks: Sequence[ToolHook] = (),
    ) -> ToolResult:
        started_at = perf_counter()
        started_iso = datetime.now(UTC).isoformat()

        tool = self._lookup_tool(tool_call)
        arguments = _parse_arguments(tool_call.arguments)
        base_context = context or ToolExecutionContext(tool_call=tool_call)
        execution_context = replace(
            base_context,
            tool_call=tool_call,
            tool_definition=tool.definition if tool is not None else None,
            arguments=arguments,
            metadata={**base_context.metadata, "started_at": started_iso},
        )
        hook_runner = ToolHookRunner(*self._hooks, *hooks) # 将多个hook集中起来，供后续在执行前、审批时和执行后调用
        permission_check = await hook_runner.before_execute(execution_context) # 返回denied_reason/approval_request/matched_rule

        if tool is None:
            result = self._failure(
                tool_call,
                f"Tool not found: {tool_call.name}",
                started_at,
            )
            return await self._complete(execution_context, result, hook_runner)

        denied_reason = await self._authorize(
            execution_context,
            hook_runner,
            permission_check,
        )
        if denied_reason is not None:
            result = self._failure(tool_call, denied_reason, started_at)
            return await self._complete(execution_context, result, hook_runner)

        result = await self._dispatch(
            tool,
            tool_call,
            execution_context,
            started_at,
        )
        return await self._complete(execution_context, result, hook_runner)


    async def _dispatch(
        self,
        tool: BaseTool,
        tool_call: ToolCall,
        context: ToolExecutionContext,
        started_at: float,
    ) -> ToolResult:
        try:
            arguments = _parse_arguments(tool_call.arguments)
        except (TypeError, ValueError) as exc:
            return self._failure(
                tool_call,
                f"Invalid arguments: {exc}",
                started_at,
            )

        try:
            async with asyncio.timeout(self._timeout_seconds):
                output = await tool.execute_with_context(arguments, context)
        except TimeoutError:
            return self._failure(
                tool_call,
                f"Tool timed out after {self._timeout_seconds:g} seconds.",
                started_at,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._failure(
                tool_call,
                f"Invalid arguments: {exc}",
                started_at,
            )
        except Exception as exc:
            return self._failure(
                tool_call,
                f"Tool execution failed: {type(exc).__name__}: {exc}",
                started_at,
            )

        serialized_output = _serialize_output(output)
        evidence_id: str | None = None
        output_sha256: str | None = None
        evidence_error: str | None = None
        if self._output_recorder is not None:
            try:
                recorded = await self._output_recorder.record(
                    context,
                    serialized_output,
                )
            except Exception as exc:
                # 工具副作用已经发生，不能因为证据落盘失败把成功伪装成失败并
                # 诱导模型重试；返回结构化告警，让调用方知道本次不可回读。
                evidence_error = f"{type(exc).__name__}: {exc}"
            else:
                if recorded is not None:
                    evidence_id = recorded.id
                    output_sha256 = recorded.sha256

        output_truncated = len(serialized_output) > self._max_output_chars
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            success=True,
            output=_truncate(serialized_output, self._max_output_chars),
            error=None,
            duration_ms=_duration_ms(started_at),
            evidence_id=evidence_id,
            output_chars=(
                len(serialized_output)
                if evidence_id is not None
                or output_truncated
                or evidence_error is not None
                else None
            ),
            output_sha256=output_sha256,
            output_truncated=True if output_truncated else None,
            evidence_error=evidence_error,
        )


    async def _authorize(
        self,
        context: ToolExecutionContext,
        hook_runner: ToolHookRunner,
        check: ToolHookDecision | None,
    ) -> str | None:
        '''
        check 无需审批 -> 返回nOne,直接放行
        check 明确禁止 -> 返回原因
        check 命中已经保存的允许规则 -> 发送审批通过通知后放行
        check 需要用户审批 -> 请求审批，等待结果
        '''
        try:
            if check is None:
                return None
            if check.denied_reason is not None:
                return check.denied_reason
            if check.approval_request is None:
                return None

            request = check.approval_request

            # 规则命中：无需询问用户，直接放行并记录命中事实。
            if check.matched_rule is not None:
                await hook_runner.on_approval_completed( #agentevent 发送审批完成通知
                    context,
                    request,
                    ApprovalDecision.APPROVED,
                    rule=check.matched_rule,
                )
                return None

            await hook_runner.on_approval_required(context, request) #agentevent 发送待审批通知
            outcome = await self._permission_hook.request_approval(
                request,
                context=context,
            )
            await hook_runner.on_approval_completed( #agentevent 发送审批完成通知
                context,
                request,
                outcome.response.decision,
                rule=outcome.rule,
            )
            return self._permission_hook.denied_reason(
                context,
                outcome.response.decision,
            )
        except Exception as exc:
            return f"Permission check failed: {type(exc).__name__}: {exc}"

    def _lookup_tool(self, tool_call: ToolCall) -> BaseTool | None:
        try:
            return self._registry.get(tool_call.name)
        except KeyError:
            return None

    def _failure(
        self,
        tool_call: ToolCall,
        message: str,
        started_at: float,
    ) -> ToolResult:
        """构造工具执行失败的结果：错误信息返回给模型，不抛异常。"""

        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            success=False,
            error=message,
            duration_ms=_duration_ms(started_at),
        )


    async def _complete(
        self,
        context: ToolExecutionContext,
        result: ToolResult,
        hook_runner: ToolHookRunner,
    ) -> ToolResult:
        """发送完成阶段并保持原始工具结果不受观察者影响。"""

        await hook_runner.after_execute(context, result)
        return result

def _serialize_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(output)


def _truncate(value: str, limit: int) -> str:
    return value[:limit]

def _duration_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000)

def main() -> None:
    import argparse
    from uuid import uuid4

    from .builtin import builtin_tool_registry

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
        registry = builtin_tool_registry(tool_names=("web_search",))
        executor = ToolExecutor(registry, timeout_seconds=args.timeout, approval_gate=ConsoleApprovalGate())
        call = ToolCall(id=str(uuid4()), name="web_search", arguments=arguments)
        result = asyncio.run(executor.execute(call, mode=AgentMode(args.mode)))
    except ValueError as exc:
        parser.exit(1, f"初始化失败：{exc}\n")
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
