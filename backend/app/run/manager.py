
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, TYPE_CHECKING

from app.model.config import AgentMode
from app.agent.events import AgentEventHandler
from app.agent.result import AgentResult
from app.checkpoint.config import CheckpointStatus
from app.context.summary import ConversationSummaryState
from app.run.config import TERMINAL_STATUSES, Run, RunStatus
from app.run.store import RunStore

if TYPE_CHECKING:
    from app.agent.runtime import AgentRuntime
    from app.checkpoint.store import CheckpointStore

logger = logging.getLogger(__name__)

class RunManager:
    """Run 生命周期管理入口（V1：start / get / list / cancel / recover）。"""

    def __init__(
        self,
        run_store: RunStore,
        checkpoint_store: CheckpointStore,
        runtime: AgentRuntime
    ) -> None:
        self._run_store = run_store
        self._checkpoint_store = checkpoint_store
        self._runtime = runtime
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        # 会话内最近一次执行完成的 AgentResult（供 CLI 读取，不持久化）。
        self._last_results: dict[str, AgentResult] = {}
    

    async def initialize(self) -> tuple[Run, ...]:
        """建表并执行启动 reconciliation，返回被修正的陈旧 Run。"""

        await self._run_store.initialize()
        return await self.reconcile()
    

    async def reconcile(self) -> tuple[Run, ...]:
        '''
        检查上一次有没有留下未正确结束的记录
        '''
        # 把所有遗留 RUNNING Checkpoint 转 INTERRUPTED
        if self._checkpoint_store is not None:
            await self._checkpoint_store.recover_running()
        
        # 创建后从未开始执行，因为没有进入agentruntime，所以没有 Checkpoint 可恢复 → 直接归入 FAILED 终态
        stale_pending = await self._run_store.list_runs(status=RunStatus.PENDING)
        reconciled: list[Run] = []
        for run in stale_pending:
            updated = await self._run_store.mark_failed(
                run.id,
                error="process restarted; pending run is marked as fail",
            )
            reconciled.append(updated)

        # 创建后开始执行，进入了agentruntime，所以可以依赖 Checkpoint 恢复状态
        stale_running = await self._run_store.list_runs(status=RunStatus.RUNNING)
        for run in stale_running:
            checkpoint = await self._checkpoint_store.get(run.id)
            if checkpoint is None:
                updated = await self._run_store.mark_failed(
                    run.id,
                    error="process restarted; run did not reach a terminal state"
                    + " (no recoverable checkpoint)",
                )
            elif checkpoint.status is CheckpointStatus.INTERRUPTED:
                updated = await self._run_store.mark_interrupted(
                    run.id,
                    error="process restarted; run did not reach a terminal state"
                    + " (recoverable checkpoint preserved)",
                )
            elif checkpoint.status is CheckpointStatus.COMPLETED:
                updated = await self._run_store.mark_completed(
                    run.id,
                    stop_reason=(
                        checkpoint.stop_reason.value
                        if checkpoint.stop_reason is not None
                        else None
                    ),
                )
            else:  # FAILED
                updated = await self._run_store.mark_failed(
                    run.id,
                    error=checkpoint.error,
                )
            reconciled.append(updated)
        return tuple(reconciled)



    async def start(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        history: tuple[Any, ...] = (),
        summary_state: ConversationSummaryState | None = None,
        event_handler: AgentEventHandler | None = None,
        recovery_run_id: str | None = None,
        recovered_from_run_id: str | None = None,
        source: str | None = None,
        source_id: str | None = None,
        scheduled_for: datetime | None = None,
        triggered_at: datetime | None = None,
        mode: AgentMode = AgentMode.DEFAULT,
    ) -> tuple[str, asyncio.Task[None]]:
    
        run = await self._run_store.create(
            conversation_id=conversation_id,
            user_message=user_message,
            recovered_from_run_id=recovered_from_run_id,
            source=source,
            source_id=source_id,
            scheduled_for=scheduled_for,
            triggered_at=triggered_at,
            mode=mode,
        )
        await self._run_store.mark_started(run.id)
        task = asyncio.create_task(
            self._execute(
                run.id,
                user_message=user_message,
                conversation_id=conversation_id,
                history=history,
                summary_state=summary_state,
                event_handler=event_handler,
                recovery_run_id=recovery_run_id,
                mode=mode,
            )
        )
        self._active_tasks[run.id] = task
        return run.id, task

    async def wait(self, run_id: str) -> Run:
        """等待 Run 执行结束并返回最终 Run 记录（幂等，可多次调用）。"""

        task = self._active_tasks.get(run_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                # 取消后的 Run 状态已由 _execute 更新为 CANCELLED。
                pass
            except Exception as exc:  # noqa: BLE001
                # 内部异常已被 _execute 标记为 failed；把信息透传给上层以便定位。
                logger.warning("Run %s 内部异常: %s", run_id, exc)
        return await self._run_store.require(run_id)

    def result(self, run_id: str) -> AgentResult | None:
        """返回本进程内最近一次执行的 AgentResult（用于 CLI 读取最终消息）。"""

        return self._last_results.get(run_id)
    
    async def get_run(self, run_id: str) -> Run | None:
        return await self._run_store.get(run_id)

    async def list_runs(
        self,
        *,
        conversation_id: str | None = None,
        status: RunStatus | str | None = None,
        limit: int = 20,
    ) -> tuple[Run, ...]:
        return await self._run_store.list_runs(
            conversation_id=conversation_id,
            status=status,
            limit=limit,
        )

    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(
            run_id
            for run_id, task in self._active_tasks.items()
            if not task.done()
        )

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------

    async def cancel(self, run_id: str) -> Run:

        run = await self._run_store.require(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        if run.status is not RunStatus.RUNNING:
            raise ValueError(
                f"cannot cancel run in state {run.status.value}"
            )
        task = self._active_tasks.get(run_id)
        if task is None or task.done():
            updated = await self._run_store.mark_cancelled(
                run_id,
                error="cancelled without active execution",
            )
            await self._cancel_pending_approvals(run_id)
            return updated
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        updated = await self._run_store.require(run_id)
        await self._cancel_pending_approvals(run_id)
        return updated

    async def interrupt(self, run_id: str) -> Run:

        run = await self._run_store.require(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        if run.status is not RunStatus.RUNNING:
            raise ValueError(
                f"cannot interrupt run in state {run.status.value}"
            )
        await self._run_store.mark_interrupted(
            run_id,
            error="interrupted by user",
        )
        task = self._active_tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        updated = await self._run_store.require(run_id)
        await self._cancel_pending_approvals(run_id)
        return updated

    async def _cancel_pending_approvals(self, run_id: str) -> None:
        """Run 取消后清理该 run 下无人等待的 PENDING approval。"""

        if self._approval_store is not None:
            await self._approval_store.cancel_pending_for_run(run_id)

    async def recover(
        self,
        run_id: str,
        *,
        history: tuple[Any, ...] = (),
        summary_state: ConversationSummaryState | None = None,
        event_handler: AgentEventHandler | None = None,
    ) -> tuple[str, asyncio.Task[None]]:

        run = await self._run_store.require(run_id)
        if run.status is not RunStatus.INTERRUPTED:
            raise ValueError(
                f"only interrupted run can be recovered: {run_id} "
                f"({run.status.value})"
            )
        checkpoint = await self._checkpoint_store.get_unrecovered(run_id)
        if checkpoint is None:
            raise ValueError(
                f"no recoverable checkpoint for run {run_id}"
            )
        return await self.start(
            checkpoint.user_message.content or run.user_message,
            conversation_id=run.conversation_id,
            history=history,
            summary_state=summary_state,
            event_handler=event_handler,
            recovery_run_id=run_id,
            recovered_from_run_id=run_id,
            # 恢复后的新 Run 沿用旧 Run 的执行模式。
            mode=run.mode,
        )

    async def _execute(
        self,
        run_id: str,
        *,
        user_message: str,
        conversation_id: str | None,
        history: tuple[Any, ...],
        summary_state: ConversationSummaryState | None,
        event_handler: AgentEventHandler | None,
        recovery_run_id: str | None,
        mode: AgentMode,
    ) -> None:
        try:
            try:
                # Runtime 直接返回结果；执行进度通过 event_handler 传递。
                result = await self._runtime.run(
                    user_message,
                    history=history,
                    conversation_id=conversation_id,
                    event_handler=event_handler,
                    summary_state=summary_state,
                    run_id=run_id,
                    recovery_run_id=recovery_run_id,
                    mode=mode,
                )
            except asyncio.CancelledError:
                current = await self._run_store.get(run_id)
                if current is None or current.status is RunStatus.RUNNING:
                    await self._run_store.mark_cancelled(
                        run_id,
                        error="cancelled by user",
                    )
                raise
            except BaseException as exc:  
                import traceback as _tb
                error = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()[-1500:]}"
                # Runtime 已把 checkpoint 标为 INTERRUPTED（保留了未决工具），
                # 此时 run 也应标为 INTERRUPTED 以便 recover，而不是 FAILED。
                recoverable = False
                if self._checkpoint_store is not None:
                    try:
                        checkpoint = await self._checkpoint_store.get(run_id)
                        recoverable = (
                            checkpoint is not None
                            and checkpoint.status is CheckpointStatus.INTERRUPTED
                        )
                    except Exception:
                        recoverable = False
                if recoverable:
                    await self._run_store.mark_interrupted(run_id, error=error)
                else:
                    await self._run_store.mark_failed(run_id, error=error)
                return

            if result is None:
                await self._run_store.mark_failed(
                    run_id,
                    error="agent produced no result",
                )
                return
            self._last_results[run_id] = result
            if result.ok:
                await self._run_store.mark_completed(
                    run_id,
                    stop_reason=result.stop_reason.value,
                )
            else:
                await self._run_store.mark_failed(
                    run_id,
                    error=(
                        result.error.message
                        if result.error is not None
                        else result.stop_reason.value
                    ),
                    stop_reason=result.stop_reason.value,
                )
        finally:
            self._active_tasks.pop(run_id, None)
