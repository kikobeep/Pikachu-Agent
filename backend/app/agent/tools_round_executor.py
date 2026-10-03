
from __future__ import annotations

from dataclasses import dataclass

from app.model.config import AgentMode, Message, MessageRole, ToolCall
from app.agent.context_injection import ContextRuntime
from app.agent.error import RepeatedToolCallError
from app.agent.result import ToolCallRecord
from app.agent.tool_hooks import AgentEventHook
from app.agent.utils import plan_id_from_output,tool_call_signature
from app.checkpoint.store import CheckpointStore
from app.skills.tools import SKILL_READ_TOOL_NAME
from app.tools.config import ToolResult
from app.tools.executor import ToolExecutor
from app.tools.config import ToolExecutionContext
from app.tools.hooks import ToolHook
from app.tools.register import TOOL_SEARCH_NAME, ToolRegistry, activated_tool_names
from app.agent.emitter import EventEmitter

@dataclass(frozen=True, slots=True)
class ToolRoundOutcome:
    """工具轮产生的记录以及需要交还给 Agent Loop 的状态。"""

    records: tuple[ToolCallRecord, ...]
    result_messages: tuple[Message, ...]
    pending_activations: frozenset[str]
    previous_signature: str | None
    repeated_count: int
    plan_created: bool
    plan_id: str | None
    repeated_error: RepeatedToolCallError | None = None

class ToolRoundExecutor:
    """执行一个模型响应中的全部结构化工具调用。"""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        executor: ToolExecutor,
        checkpoint_store: CheckpointStore | None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._checkpoint_store = checkpoint_store

    async def execute(
        self,
        tool_calls: tuple[ToolCall, ...],
        *,
        run_id: str,
        conversation_id: str | None,
        user_input: str,
        step: int,
        mode: AgentMode,
        round_index: int,
        closing_can_deliver: bool,
        activated_tools: set[str],
        context_session: ContextRuntime,
        previous_signature: str | None,
        repeated_count: int,
        emitter: EventEmitter,
        hook: AgentEventHook,
    ) -> ToolRoundOutcome:
        """执行工具轮；重复调用错误也连同已完成的前序记录返回。"""

        records: list[ToolCallRecord] = []
        result_messages: list[Message] = []
        pending_activations: set[str] = set()
        plan_created = False
        plan_id: str | None = None

        if self._checkpoint_store is not None:
            await self._checkpoint_store.before_tools(
                run_id,
                step=step,
                tool_calls=tool_calls,
            )

        for tool_call in tool_calls:
            signature = tool_call_signature(tool_call)
            if signature == previous_signature:
                repeated_count += 1
            else:
                previous_signature = signature
                repeated_count = 1
            if repeated_count >= 3:
                return ToolRoundOutcome(
                    records=tuple(records),
                    result_messages=tuple(result_messages),
                    pending_activations=frozenset(pending_activations),
                    previous_signature=previous_signature,
                    repeated_count=repeated_count,
                    plan_created=plan_created,
                    plan_id=plan_id,
                    repeated_error=RepeatedToolCallError(tool_call.name),
                )


            context = ToolExecutionContext(
                run_id=run_id,
                conversation_id=conversation_id,
                user_input=user_input,
                step=step,
                tool_call=tool_call,
                metadata={
                    "active_skill_names": context_session.active_skill_names
                },
                mode=mode,
            )
            result = await self._execute_one(
                tool_call,
                context=context,
                hook=hook,
                mode=mode,
                closing_can_deliver=closing_can_deliver,
                activated_tools=activated_tools
            )

            if self._checkpoint_store is not None:
                await self._checkpoint_store.complete_tool(run_id, result)
            '''
            plan_create / plan_update
                ↓
            任务存入 Task Store
                ↓
            TaskContextProvider 读取当前会话绑定的活动任务
                ├─ 渲染任务状态，加入模型上下文
                └─ 提取标题、进行中步骤，补充记忆召回 query
            '''
            if (
                mode is AgentMode.PLAN
                and result.success
                and tool_call.name in ("plan_create", "plan_update")
            ):
                plan_created = True
                extracted_plan_id = plan_id_from_output(result.output)
                if extracted_plan_id:
                    plan_id = extracted_plan_id

            records.append(
                ToolCallRecord(
                    round_index=round_index,
                    tool_call=tool_call,
                    result=result,
                )
            )
            result_messages.append(self._result_message(result))
            if tool_call.name == TOOL_SEARCH_NAME and result.success:
                pending_activations.update(
                    name
                    for name in activated_tool_names(result.output)
                    if self._registry.is_on_demand(name)
                )
            if tool_call.name == SKILL_READ_TOOL_NAME and result.success:
                await context_session.activate_skill(
                    result,
                    emitter=emitter,
                    step=step,
                )

        return ToolRoundOutcome(
            records=tuple(records),
            result_messages=tuple(result_messages),
            pending_activations=frozenset(pending_activations),
            previous_signature=previous_signature,
            repeated_count=repeated_count,
            plan_created=plan_created,
            plan_id=plan_id,
        )


    async def _execute_one(
        self,
        tool_call: ToolCall,
        *,
        context: ToolExecutionContext,
        hook: ToolHook,
        mode: AgentMode,
        closing_can_deliver: bool,
        activated_tools: set[str]
    ) -> ToolResult:
        """执行一次工具调用，并在执行层落实模式与 Closing 边界。"""

        rejection = self._rejection_reason(
            tool_call,
            mode=mode,
            closing_can_deliver=closing_can_deliver,
            activated_tools=activated_tools
        )
        if rejection is not None:
            await hook.before_execute(context)
            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=rejection,
                duration_ms=0,
            )
            await hook.after_execute(context, result)
            return result
        try:
            return await self._executor.execute(
                tool_call,
                context=context,
                hooks=(hook,),
            )
        except Exception as exc:
            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=0.0,
            )
            await hook.after_execute(context, result)
            return result
    

    def _rejection_reason(
        self,
        tool_call: ToolCall,
        *,
        mode: AgentMode,
        closing_can_deliver: bool,
        activated_tools: set[str]
    ) -> str | None:
        if closing_can_deliver and not (
            self._registry.is_closing_allowed(tool_call.name, mode)
            and (
                not self._registry.is_on_demand(tool_call.name)
                or tool_call.name in activated_tools
            )
        ):
            return (
                "Tool is not allowed during budget closing "
                "(delivery tools only)."
            )
        if mode is AgentMode.PLAN and not self._registry.is_allowed_for_mode(
            tool_call.name,
            mode,
        ):
            return (
                "Tool is not allowed in plan mode "
                "(read-only / planning tools only)."
            )
        if not self._registry.is_available_for_mode(
            tool_call.name,
            mode,
            activated_names=activated_tools,
        ):
            return "Deferred tool is not active. Call tool_search first."
        return None


    @staticmethod
    def _result_message(result: ToolResult) -> Message:
        return Message(
            role=MessageRole.TOOL,
            name=result.tool_name,
            tool_call_id=result.tool_call_id,
            content=result.model_dump_json(exclude_none=True),
        )

'''
    AgentLoop 收到模型回复
│
│  assistant_message.tool_calls
│  例如：[读取文件、执行测试]
│
▼
ToolRoundExecutor.execute(tool_calls, ...)
│
├─ 1. 写入 checkpoint.before_tools()
│     记录这一轮待执行的工具调用
│
└─ 2. 按顺序遍历每一个 tool_call
      │
      ├─ 检查是否连续重复调用
      │     达到阈值 → 返回重复调用错误，停止这一轮
      │
      ├─ 创建 ToolExecutionContext
      │     包含 run_id、conversation_id、step、
      │     当前工具调用、已激活 Skill 名称等
      │
      ▼
    ToolRoundExecutor._execute_one()
      │
      ├─ 检查当前运行状态是否允许调用
      │     · 是否符合 Plan 模式限制？
      │     · 预算收尾时是否允许这个工具？
      │     · 按需工具是否已激活？
      │
      ├─ 不允许
      │     → 发送开始／结束事件
      │     → 返回失败 ToolResult
      │
      └─ 允许
            │
            ▼
          ToolExecutor.execute(tool_call, context, hooks)
            │
            ├─ 3. 准备执行信息
            │     · 从 Registry 查找工具对象
            │     · 整理参数，供权限检查使用
            │     · 补齐 tool_definition、开始时间等上下文
            │     · 汇总权限、日志和本次额外传入的 Hook
            │
            ├─ 4. hook_runner.before_execute()
            │     · PermissionHook：检查权限和已保存规则，这里的权限指的是ToolPermission.FORBIDDEN->deny，在rule store里搜索，如果没搜到->approval
            │     · 其他 Hook：记录日志、发送工具开始事件等
            │     → 得到 permission_check
            │
            ├─ 工具不存在 → 构造失败 ToolResult
            │
            ▼
          ToolExecutor._authorize(permission_check)
            │
            ├─ 无需审批 → 放行
            │
            ├─ 明确拒绝 → 返回拒绝原因
            │
            ├─ 命中允许规则
            │     → 通知审批通过
            │     → 放行
            │
            └─ 需要人工审批
                  │
                  ├─ on_approval_required()
                  │     通知：正在等待审批
                  │
                  ├─ PermissionHook.request_approval()
                  │     → ApprovalGate 获取审批决定
                  │     → 如批准范围为 Run／会话，尝试保存规则
                  │
                  ├─ on_approval_completed()
                  │     通知：审批结果
                  │
                  └─ 批准 → 放行
                     拒绝 → 返回拒绝原因
            │
            ├─ 未获授权 → 构造失败 ToolResult
            │
            └─ 获得授权
                  │
                  ▼
                ToolExecutor._dispatch()
                  │
                  ├─ 5. 解析工具参数
                  │     参数不合法 → 失败 ToolResult
                  │
                  ├─ 6. 在超时限制内调用工具
                  │
                  │     tool.execute_with_context(arguments, context)
                  │       │
                  │       ├─ 子类重写了 → 执行子类方法
                  │       └─ 没重写 → 默认转调 execute(arguments)
                  │
                  │     超时／抛出异常 → 失败 ToolResult
                  │
                  ├─ 7. 工具正常返回后，序列化输出
                  │
                  ├─ 8. EvidenceRecorder 保存原始输出
                  │     满足记录条件时写入 EvidenceStore
                  │     得到 evidence_id
                  │
                  └─ 9. 构造成功 ToolResult
                        · output：可能截断后的输出
                        · evidence_id：原始输出的查询入口
                        · duration_ms 等执行信息
            │
            ▼
          ToolExecutor._complete()
            │
            ├─ 调用 after_execute() Hooks
            │     记录日志、发送工具完成事件等
            │
            └─ 返回 ToolResult
      │
      ▼
    回到 ToolRoundExecutor
      │
      ├─ 10. 根据结果更新状态
      │      · 电脑操作防重复／验证状态
      │      · checkpoint.complete_tool()
      │      · Plan 模式下的任务状态标记
      │
      ├─ 11. 保存本次调用记录
      │      ToolCallRecord(tool_call, result, ...)
      │
      ├─ 12. 将结果转换成 TOOL 消息
      │      使用 tool_call_id 对应模型发起的调用
      │
      ├─ 13. 处理成功调用的特殊结果
      │      · tool_search → 收集待激活工具名称
      │      · skill_read → 激活读取到的 Skill
      │
      └─ 继续执行下一个 tool_call
│
▼
返回 ToolRoundOutcome 给 AgentLoop
│
├─ 工具结果消息追加到 messages
├─ 调用记录加入 tool_calls、tool_rounds
├─ 待激活工具加入 activated_tools
├─ 更新重复调用等状态
│
▼
进入下一个 Step
重新准备上下文和工具列表，请求模型
模型根据工具结果决定继续调用工具，还是回答用户
    '''
