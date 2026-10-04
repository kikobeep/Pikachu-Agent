
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator,field_validator
from datetime import UTC, datetime

from app.model.config import AgentMode, Message, MessageRole
from app.agent.events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
    MultiEventHandlers,
)
from app.agent.emitter import EventEmitter
from app.agent.result import AgentError, AgentResult, AgentStopReason
from app.context.summary_store import ConversationSummaryStore
from app.context.manager import ContextDecision, ContextManager
from app.conversation.store import ConversationStore
from app.run.config import Run, RunStatus
from app.run.manager import RunManager
from app.trace.store import TraceStore, TraceEventHandler
from app.tools.config import ToolDefinition



class ConversationSource(StrEnum):
    MANUAL = "manual"
    AUTOMATION = "automation"
    # 评测脚本的首轮盲测 Run；只允许沉淀首轮可见信息。
    EVAL_INITIAL = "eval_initial"
    # 评测脚本的 feedback 修复轮：这类 Run 已被喂入真实测试错误，
    # 其经验隐含「已知错误信息」前提，不应沉淀为通用 skill。
    EVAL_FEEDBACK = "eval_feedback"

class TriggerContext(BaseModel):
    """一次输入投递的触发来源与调度元数据。"""

    model_config = ConfigDict(extra="forbid")

    source: ConversationSource
    automation_id: str | None = None
    scheduled_for: datetime | None = None
    triggered_at: datetime | None = None

    @field_validator("scheduled_for", "triggered_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trigger datetimes must include timezone information")
        return value.astimezone(UTC)

@dataclass
class DispatchResult:
    """一次 Conversation 输入投递的完整结果。"""

    run: Run
    result: AgentResult
    trigger: TriggerContext
    conversation_id: str | None

class ConversationService:
    def __init__(
        self,
        conversation_store: ConversationStore,
        run_manager: RunManager,
        trace_store: TraceStore,
        *,
        summary_store: ConversationSummaryStore | None = None
    ) -> None:
        self._conversation_store = conversation_store
        self._run_manager = run_manager
        self._trace_store = trace_store
        self._summary_store = summary_store
        self._locks : dict[str,Any] = {}

    # 每一个conversation都需要上锁
    def _lock(self,conversation_id):
        lock = self._locks.get(conversation_id)
        if lock is None: 
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock
    
    
    async def dispatch(
        self,
        *,
        conversation_id: str | None,
        content: str,
        trigger: TriggerContext | None = None,
        event_handler: AgentEventHandler | None = None,
        on_run_started: Any | None = None,
        mode: AgentMode = AgentMode.DEFAULT,
    ) -> DispatchResult:

        trigger = trigger or TriggerContext(
            source=ConversationSource.MANUAL
        )
        async with self._lock(conversation_id):
            return await self._dispatch_locked(
                conversation_id=conversation_id,
                content=content,
                trigger=trigger,
                event_handler=event_handler,
                on_run_started=on_run_started,
                mode=mode,
            )

    async def compact(
        self,
        *,
        conversation_id: str,
        context_manager: ContextManager,
        tools: tuple[ToolDefinition, ...] = (),
        provider: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        event_handler: AgentEventHandler | None = None,
    ) -> ContextDecision:
        """手动压缩会话上下文，但不创建 Run 或追加聊天消息。"""

        async with self._lock(conversation_id):
            history = tuple(
                await self._conversation_store.load_messages(conversation_id)
            )
            summary_state = (
                await self._summary_store.load(conversation_id)
                if self._summary_store is not None
                else None
            )
            decision = await context_manager.prepare(
                history,
                tools=tools,
                model=model,
                provider=provider,
                max_output_tokens=max_output_tokens,
                history_count=len(history),
                summary_state=summary_state,
                force_compaction=True,
            )
            if (
                self._summary_store is not None
                and decision.summary_state is not None
                and decision.summary_state != summary_state
            ):
                await self._summary_store.save(
                    conversation_id,
                    decision.summary_state,
                )
            if event_handler is not None:
                summary_text = None
                if decision.summary_state is not None:
                    summary_text = decision.summary_state.summary.render_markdown()
                emitter = EventEmitter(
                    handler=event_handler,
                    run_id=f"compact-{conversation_id[:24]}",
                    conversation_id=conversation_id,
                )
                await emitter.emit(
                    AgentEventType.CONTEXT_COMPACTED,
                    compaction_stage=decision.compaction_stage.value,
                    compacted_tool_results=decision.compacted_tool_results,
                    removed_tool_rounds=decision.removed_tool_rounds,
                    reached_target=decision.reached_target,
                    needs_next_compaction_stage=decision.needs_next_compaction_stage,
                    summary_updated=decision.summary_updated,
                    summarized_conversation_blocks=decision.summarized_conversation_blocks,
                    summary_usage=decision.summary_usage,
                    summary_provider=decision.summary_provider,
                    summary_model=decision.summary_model,
                    summary_duration_ms=decision.summary_duration_ms,
                    summary_error=decision.summary_error,
                    summary_text=summary_text,
                )
            return decision
    
    async def _dispatch_locked(
        self,
        *,
        conversation_id: str | None,
        content: str,
        trigger: TriggerContext,
        event_handler: AgentEventHandler | None,
        on_run_started: Any | None,
        mode: AgentMode,
    ) -> DispatchResult:
        '''
        读取旧对话，执行这次输入，再保存新对话
        '''
        # 1) 读取旧对话
        history: tuple[Any, ...] = ()
        if conversation_id is not None:
            history = tuple(
                await self._conversation_store.load_messages(conversation_id)
            )
        summary_state = (
            await self._summary_store.load(conversation_id)
            if self._summary_store is not None and conversation_id is not None
            else None
        )

        # 2) trace handle + cli hander。
        trace_handler = TraceEventHandler(self._trace_store)
        event_handlers: list[AgentEventHandler] = [trace_handler] # cli handler

        if event_handler is not None:
            event_handlers.append(event_handler)

       
        if len(event_handlers) == 1:
            handler = event_handlers[0]
        else:
            handler = MultiEventHandlers(*event_handlers)
       

        # 3) 执行这次输入，启动 Run
        run_id, _ = await self._run_manager.start(
            content,
            conversation_id=conversation_id,
            history=history,
            summary_state=summary_state,
            event_handler=handler,
            source=trigger.source.value,
            source_id=trigger.automation_id,
            scheduled_for=trigger.scheduled_for,
            triggered_at=trigger.triggered_at,
            mode=mode,
        )
        try:
            run = await self._run_manager.wait(run_id)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl-C 可能表现为 KeyboardInterrupt，也可能表现为 asyncio
            # 取消当前 CLI 主任务；两种情况都必须保留可恢复的 Run。
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            await self._run_manager.cancel(run_id)
            run = await self._run_manager.wait(run_id)
        result = self._run_manager.result(run_id)

        if result is None:
            if run.status is RunStatus.CANCELLED:
                cancelled_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "Run cancelled：已停止，未生成最终回复。"
                        "（本轮未完成的内容不会显示）"
                    ),
                )
                result = AgentResult(
                    run_id=run_id,
                    final_message=cancelled_message,
                    messages=(
                        *history,
                        Message(role=MessageRole.USER, content=content),
                        cancelled_message,
                    ),
                    steps=0,
                    stop_reason=AgentStopReason.CANCELLED,
                )
                await handler.handle(
                    AgentEvent(
                        run_id=run_id,
                        conversation_id=conversation_id,
                        sequence=0,
                        type=AgentEventType.AGENT_CANCELLED,
                        message=cancelled_message,
                        stop_reason=AgentStopReason.CANCELLED,
                        result=result,
                    )
                )
            elif run.status is RunStatus.INTERRUPTED:
                interrupted_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "Run interrupted：已暂停，可从断点继续。"
                        "（点击 Recover 从保存的中断点恢复）"
                    ),
                )
                result = AgentResult(
                    run_id=run_id,
                    final_message=interrupted_message,
                    messages=(
                        *history,
                        Message(role=MessageRole.USER, content=content),
                        interrupted_message,
                    ),
                    steps=0,
                    stop_reason=AgentStopReason.INTERRUPTED,
                )
                await handler.handle(
                    AgentEvent(
                        run_id=run_id,
                        conversation_id=conversation_id,
                        sequence=0,
                        type=AgentEventType.AGENT_FAILED,
                        message=interrupted_message,
                        stop_reason=AgentStopReason.INTERRUPTED,
                        result=result,
                    )
                )
            elif run.status is RunStatus.FAILED:
                # RunManager 在执行过程中捕获到未处理异常并把 run 标记为 failed
                # 但仍未构造 AgentResult。此时合成一个失败响应，避免 CLI 直接抛错。
                import traceback as _tb
                error_text = run.error or "RunManager 未返回最终 AgentResult"
                failed_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        f"[Run failed] {error_text}\n\n"
                        "本次 Run 因内部错误结束，可稍后重试或调整输入。\n\n"
                        f"TRACEBACK:\n{_tb.format_exc()[-2000:]}"
                    ),
                )
                result = AgentResult(
                    run_id=run_id,
                    final_message=failed_message,
                    messages=(
                        *history,
                        Message(role=MessageRole.USER, content=content),
                        failed_message,
                    ),
                    steps=0,
                    stop_reason=AgentStopReason.MODEL_ERROR,
                    error=AgentError(
                        type=RunStatus.FAILED.value,
                        message=error_text,
                    ),
                )
                await handler.handle(
                    AgentEvent(
                        run_id=run_id,
                        conversation_id=conversation_id,
                        sequence=0,
                        type=AgentEventType.AGENT_FAILED,
                        message=failed_message,
                        stop_reason=AgentStopReason.MODEL_ERROR,
                        result=result,
                    )
                )
            else:
                raise RuntimeError("RunManager 未返回最终 AgentResult")

        # 4) 保存执行后的会话：把新的完整 history 写回 ConversationStore（保存最新 Summary）。
        if conversation_id is not None:
            await self._conversation_store.replace_messages(
                conversation_id,
                result.messages,
            )
        if (
            self._summary_store is not None
            and conversation_id is not None
            and result.summary_state is not None
        ):
            await self._summary_store.save(
                conversation_id,
                result.summary_state,
            )

        return DispatchResult(
            run=run,
            result=result,
            trigger=trigger,
            conversation_id=conversation_id,
        )


'''
① chat：接收用户输入
│
├─ input() 得到“帮我看看项目的测试为什么失败”
├─ 判断不是 /new、/use 等管理命令
└─ 调用 _send_message()
       ↓
   conversation_service.dispatch(
       conversation_id=当前会话ID,
       content=用户输入,
       event_handler=CLI进度打印器,
   )
       │
       ▼
② ConversationService：准备会话数据
│
├─ 获取当前会话的锁
│   防止同一会话的两次输入同时修改历史
│
├─ 从 ConversationStore 读取最新历史
├─ 从 SummaryStore 读取已有摘要
├─ 组合事件处理器：Trace记录 + CLI进度打印
│
└─ 调用 run_manager.start(
       当前输入、历史、摘要、事件处理器……
   )
       │
       ▼
③ RunManager：创建并启动本次 Run
│
├─ RunStore.create()：创建 PENDING 记录
├─ mark_started()：改成 RUNNING
│
├─ asyncio.create_task(self._execute(...))
│   创建执行任务，保存到 _active_tasks[run_id]
│
└─ 返回 run_id 和 task
       │
       ├─ ConversationService 调用 wait(run_id)
       │   暂停在这里，等待执行结束
       │
       └─ 后台 task 执行 _execute()
              ↓
          result = await runtime.run(...)
              │
              ▼
④ AgentRuntime：组织本次运行
│
├─ 创建 EventEmitter
│   给事件附上 run_id、conversation_id、序号
│
├─ 如需恢复，读取旧 checkpoint
├─ 创建本次 Run 的 checkpoint
│
└─ result = await loop.run(...)
       │
       ▼
⑤ AgentLoop：模型与工具循环
│
├─ 初始化本次 Run 的状态和 ContextSession
│
└─ for step ...
    │
    ├─ 检查步数、工具轮次和累计预算
    │
    ├─ 准备本次模型请求
    │   · 聊天历史和当前输入
    │   · 记忆、Skill、Task 等上下文
    │   · 当前可用工具定义
    │   · 预算提醒或收尾指令
    │
    ├─ 尝试复用稳定前缀
    ├─ ContextManager 检查输入大小
    │   必要时压缩工具结果、生成历史摘要
    ├─ 再次检查预算
    │
    ├─ 调用模型
    │
    ├─ 模型要求调用工具
    │      ↓
    │   ToolRoundExecutor
    │      ↓
    │   ToolExecutor：权限检查、审批、执行工具
    │      ↓
    │   将工具结果加入 messages
    │   更新工具激活、Skill等状态
    │      ↓
    │   进入下一个 Step
    │
    ├─ 模型给出可修正的异常回复
    │   如空回复 → 在允许范围内添加提示并重试
    │
    └─ 模型给出最终回答，或遇到终止条件
           ↓
       返回 AgentResult
'''
