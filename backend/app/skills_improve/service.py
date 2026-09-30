"""Skill improvement orchestration.

This module is deliberately an orchestration layer.  It does not decide how a
model mines patterns or writes a candidate; those responsibilities are injected
as ``miner`` and ``distiller`` collaborators.  A improving sample is anchored
at one Run, while the conversation is used only for the user request and
feedback relevant to that Run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.agent.events import AgentEvent
from app.conversation.store import ConversationStore
from app.model.config import Message, MessageRole
from app.model.registry import ModelAdapterRegistry
from app.run.config import Run, RunStatus
from app.run.outcome_store import RunOutcomeStore
from app.run.store import RunStore
from app.skills.store import SkillStore
from app.trace.store import TraceStore

from .config import SkillImprovingSettings

logger = logging.getLogger("sidekick.skills_improve.service")


@dataclass(frozen=True, slots=True)
class ImprovingSample:
    """One completed Run projected for skill improving."""

    run: Run
    user_request: str
    conversation_messages: tuple[Message, ...] = ()
    user_feedback: tuple[str, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    evidence: str = ""
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class SkillImproveResult:
    """Result of one improving attempt."""

    triggered: bool = False
    skipped_reason: str | None = None
    scanned_run_count: int = 0
    sample_count: int = 0
    cluster_count: int = 0
    candidate_count: int = 0
    processed_run_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunImprovingWatermark:
    """持久化 Run 改进水位，支持失败批次重试。"""

    processed_run_ids: tuple[str, ...] = ()
    processing_run_ids: tuple[str, ...] = ()
    last_error: str | None = None


class PatternMiner(Protocol):
    async def mine(self, samples: tuple[ImprovingSample, ...]) -> Any: ...


class ProcedureDistiller(Protocol):
    async def distill(
        self,
        cluster: Any,
        *,
        samples: tuple[ImprovingSample, ...],
        catalog: Sequence[Any],
        skill_loader: Callable[[str], Awaitable[Any | None]],
    ) -> Any: ...


class EventSelector(Protocol):
    def select(
        self,
        run: Run,
        events: tuple[AgentEvent, ...],
    ) -> tuple[AgentEvent, ...]: ...


class EvidenceBuilder(Protocol):
    def build(
        self,
        run: Run,
        events: tuple[AgentEvent, ...],
    ) -> str: ...


class CandidateStore(Protocol):
    """Minimal persistence contract required by this service."""

    async def create(self, candidate: Any) -> Any: ...
    async def accept(self, candidate_id: str) -> Any: ...


class SkillImproveService:

    def __init__(
        self,
        run_store: RunStore,
        trace_store: TraceStore,
        skill_store: SkillStore,
        candidate_store: CandidateStore,
        registry: ModelAdapterRegistry,
        *,
        settings: SkillImprovingSettings,
        conversation_store: ConversationStore | None = None,
        outcome_store: RunOutcomeStore | None = None,
        miner: PatternMiner | None = None,
        distiller: ProcedureDistiller | None = None,
        selector: EventSelector | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        watermark_path: str | Path | None = None,
    ) -> None:
        self.run_store = run_store
        self.trace_store = trace_store
        self.skill_store = skill_store
        self.candidate_store = candidate_store
        self.conversation_store = conversation_store
        self.outcome_store = outcome_store
        self.registry = registry
        self.miner = miner
        self.distiller = distiller
        self.selector = selector
        self.evidence_builder = evidence_builder
        self.settings = settings
        self.batch_size = settings.batch_size
        self.max_runs_per_scan = settings.max_runs_per_scan
        self.max_conversation_messages = settings.max_conversation_messages
        self.max_feedback_chars = settings.max_feedback_chars
        self.auto_accept = settings.auto_accept
        self.improve_method = settings.improve_method
        self._watermark_path = self._resolve_watermark_path(
            watermark_path or (Path(settings.data_dir) / "watermark.json")
        )
        self._watermark_lock = asyncio.Lock()

    async def maybe_run_improving(self) -> SkillImproveResult:
        """Scan completed Runs and run at most one improving batch.

        The processing batch is persisted as ``processing`` before model calls.
        It moves to ``processed`` only after all stages succeed; failures keep
        the batch processing so a later invocation can retry it.
        """

        if self.miner is None or self.distiller is None:
            return SkillImproveResult(skipped_reason="pipeline_not_configured")

        completed_runs = await self.run_store.list_runs(
            status=RunStatus.COMPLETED,
            limit=self.max_runs_per_scan,
        )
        watermark = await self._load_watermark()
        processed_run_ids = set(watermark.processed_run_ids)
        unprocessed_runs = tuple(
            run for run in completed_runs if run.id not in processed_run_ids
        )
        # 排除评测脚本的 feedback 修复轮（ConversationSource.EVAL_FEEDBACK）：
        # 这类 Run 已被喂入真实测试错误，其经验隐含「已知错误信息」前提，
        # 盲测时不成立，沉淀成 skill 反而会诱导模型去搜它看不到的东西。
        # 只对评测脚本显式打上的这个来源标记生效，不影响日常会话。
        unprocessed_runs = tuple(
            run for run in unprocessed_runs if run.source != "eval_feedback"
        )
        if self.outcome_store is not None:
            outcome_runs: list[Run] = []
            for run in unprocessed_runs:
                outcome = await self.outcome_store.get(run.id)
                # 评测首轮没有明确结果时不参与本批；普通会话仍可走原有流程。
                if run.source == "eval_initial" and outcome is None:
                    continue
                outcome_runs.append(run)
            unprocessed_runs = tuple(outcome_runs)
        if not watermark.processing_run_ids and len(unprocessed_runs) < self.batch_size:
            return SkillImproveResult(
                skipped_reason="batch_not_ready",
                scanned_run_count=len(completed_runs),
            )

        # 优先恢复上次未完成的批次；没有恢复批次时才从新的 Run 中截取一批。
        batch_run_ids = watermark.processing_run_ids or tuple(
            run.id for run in unprocessed_runs[: self.batch_size]
        )
        runs_by_id = {run.id: run for run in completed_runs}
        if watermark.processing_run_ids:
            for run_id in batch_run_ids:
                if run_id not in runs_by_id:
                    run = await self.run_store.get(run_id)
                    if run is not None and run.status is RunStatus.COMPLETED:
                        runs_by_id[run_id] = run
        batch_runs = tuple(
            runs_by_id[run_id]
            for run_id in batch_run_ids
            if run_id in runs_by_id
        )
        if len(batch_runs) < self.batch_size:
            return SkillImproveResult(
                skipped_reason="processing_runs_not_available",
                scanned_run_count=len(completed_runs),
            )
        await self._save_watermark(
            RunImprovingWatermark(
                processed_run_ids=tuple(sorted(processed_run_ids)),
                processing_run_ids=tuple(run.id for run in batch_runs),
            )
        )
        samples: list[ImprovingSample] = []
        errors: list[str] = []
        for run in batch_runs:
            try:
                samples.append(await self._build_sample(run))
            except Exception as exc:  # isolate one corrupt Run from the batch
                errors.append(f"{run.id}: {type(exc).__name__}: {exc}")
                logger.warning("failed to build improving sample run=%s", run.id)

        if not samples:
            message = "; ".join(errors) or "no improving samples"
            await self._save_watermark(
                RunImprovingWatermark(
                    processed_run_ids=tuple(sorted(processed_run_ids)),
                    processing_run_ids=tuple(run.id for run in batch_runs),
                    last_error=message,
                )
            )
            return SkillImproveResult(
                triggered=True,
                scanned_run_count=len(completed_runs),
                processed_run_ids=(),
                errors=tuple(errors) or ("no improving samples",),
            )

        try:
            if self.improve_method == "multi_teacher":
                # This mode intentionally skips explicit clustering: the two
                # teachers analyze all successful/failed traces and the final
                # distiller decides what can be merged into one Skill.
                clusters = ()
            else:
                mining = await self.miner.mine(tuple(samples))
                clusters = tuple(getattr(mining, "clusters", ()) or ())
        except Exception as exc:
            error = f"pattern mining: {type(exc).__name__}: {exc}"
            await self._save_watermark(
                RunImprovingWatermark(
                    processed_run_ids=tuple(sorted(processed_run_ids)),
                    processing_run_ids=tuple(run.id for run in batch_runs),
                    last_error=error,
                )
            )
            return SkillImproveResult(
                triggered=True,
                scanned_run_count=len(completed_runs),
                sample_count=len(samples),
                errors=tuple(errors) + (error,),
            )

        # 去重目录：已有正式 skill + 本批已生成的候选，一起喂给后续蒸馏。
        catalog = list(await self.skill_store.load_metadata())
        seen_names = {
            getattr(item, "name", "")
            for item in catalog
            if getattr(item, "name", "")
        }
        created_candidate_count = 0
        if self.improve_method == "multi_teacher" and hasattr(
            self.distiller, "distill_multi_teacher"
        ):
            try:
                distillation_result = await self.distiller.distill_multi_teacher(
                    samples=tuple(samples), catalog=tuple(catalog)
                )
                candidate = getattr(distillation_result, "candidate", None)
                if candidate is not None and candidate.name not in seen_names:
                    await self.candidate_store.create(candidate)
                    if self.auto_accept:
                        await self.candidate_store.accept(candidate.id)
                    created_candidate_count = 1
            except Exception as exc:
                errors.append(f"multi_teacher: {type(exc).__name__}: {exc}")
        for cluster in (() if self.improve_method == "multi_teacher" else clusters):
            try:
                distillation_result = await self.distiller.distill(
                    cluster,
                    samples=tuple(sample for sample in samples if self._sample_in_cluster(sample, cluster)),
                    catalog=tuple(catalog),
                    skill_loader=self.skill_store.load,
                )
                candidate = getattr(distillation_result, "candidate", None)
                if candidate is None:
                    continue
                if candidate.name in seen_names:
                    # 硬去重：名字已存在（已有 skill 或本批已生成）则丢弃。
                    logger.info(
                        "skip duplicate candidate name=%s cluster=%s",
                        candidate.name,
                        getattr(cluster, "id", "cluster"),
                    )
                    continue
                await self.candidate_store.create(candidate)
                if self.auto_accept:
                    await self.candidate_store.accept(candidate.id)
                created_candidate_count += 1
                seen_names.add(candidate.name)
                catalog.append(candidate)
            except Exception as exc:
                errors.append(
                    f"{getattr(cluster, 'id', 'cluster')}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if errors:
            await self._save_watermark(
                RunImprovingWatermark(
                    processed_run_ids=tuple(sorted(processed_run_ids)),
                    processing_run_ids=tuple(run.id for run in batch_runs),
                    last_error="; ".join(errors),
                )
            )
        else:
            processed_run_ids.update(run.id for run in batch_runs)
            await self._save_watermark(
                RunImprovingWatermark(
                    processed_run_ids=tuple(sorted(processed_run_ids)),
                    processing_run_ids=(),
                )
            )
        return SkillImproveResult(
            triggered=True,
            scanned_run_count=len(completed_runs),
            sample_count=len(samples),
            cluster_count=len(clusters),
            candidate_count=created_candidate_count,
            processed_run_ids=(
                tuple(run.id for run in batch_runs) if not errors else ()
            ),
            errors=tuple(errors),
        )

    @staticmethod
    def _resolve_watermark_path(path: str | Path | None) -> Path:
        if path is not None:
            return Path(path).expanduser().resolve()
        return Path(".sidekick/skill-improve-watermark.json").resolve()

    async def _load_watermark(self) -> RunImprovingWatermark:
        async with self._watermark_lock:
            return await asyncio.to_thread(self._read_watermark)

    async def _save_watermark(self, watermark: RunImprovingWatermark) -> None:
        async with self._watermark_lock:
            await asyncio.to_thread(self._write_watermark, watermark)

    def _read_watermark(self) -> RunImprovingWatermark:
        try:
            payload = json.loads(self._watermark_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return RunImprovingWatermark()
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("invalid skill improving watermark: %s", exc)
            return RunImprovingWatermark()
        if not isinstance(payload, dict):
            return RunImprovingWatermark()
        return RunImprovingWatermark(
            processed_run_ids=_string_tuple(payload.get("processed_run_ids")),
            processing_run_ids=_string_tuple(
                payload.get(
                    "processing_run_ids",
                    payload.get("inflight_run_ids"),
                )
            ),
            last_error=(
                payload.get("last_error")
                if isinstance(payload.get("last_error"), str)
                else None
            ),
        )

    def _write_watermark(self, watermark: RunImprovingWatermark) -> None:
        self._watermark_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._watermark_path.with_name(
            f".{self._watermark_path.name}.tmp"
        )
        payload = json.dumps(
            {
                "processed_run_ids": list(watermark.processed_run_ids),
                "processing_run_ids": list(watermark.processing_run_ids),
                "last_error": watermark.last_error,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            with temporary.open("w", encoding="utf-8") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._watermark_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    async def build_sample(self, run_id: str) -> ImprovingSample:
        """Build one Run-anchored sample for tests or explicit reprocessing."""

        run = await self.run_store.get(run_id)
        if run is None:
            raise KeyError(f"Run 不存在：{run_id}")
        if run.status is not RunStatus.COMPLETED:
            raise ValueError(f"Run 尚未完成：{run_id}")
        return await self._build_sample(run)

    async def _build_sample(self, run: Run) -> ImprovingSample:
        events = tuple(await self.trace_store.load_events(run.id))
        selected_events = (
            self.selector.select(run, events)
            if self.selector is not None
            else events
        )
        evidence = (
            self.evidence_builder.build(run, selected_events)
            if self.evidence_builder is not None
            else _render_events(selected_events)
        )

        # 评测首轮与 feedback 轮共享 conversation。首轮样本只能使用
        # Run 自己的输入，不能回读 conversation，否则未来新增字段使用
        # conversation_messages/user_feedback 时会把 feedback 带入蒸馏。
        is_eval_initial = run.source == "eval_initial"
        messages: tuple[Message, ...] = ()
        if (
            not is_eval_initial
            and self.conversation_store is not None
            and run.conversation_id
        ):
            messages = await self.conversation_store.load_messages(
                run.conversation_id
            )
            messages = messages[-self.max_conversation_messages :]

        feedback = (
            ()
            if is_eval_initial
            else _extract_user_feedback(messages, self.max_feedback_chars)
        )
        outcome = (
            await self.outcome_store.get(run.id)
            if self.outcome_store is not None
            else None
        )
        user_request = run.user_message.strip()
        if not user_request:
            user_request = next(
                (
                    message.content.strip()
                    for message in messages
                    if message.role is MessageRole.USER
                    and message.content
                    and message.content.strip()
                ),
                "",
            )
        return ImprovingSample(
            run=run,
            user_request=user_request,
            conversation_messages=messages,
            user_feedback=feedback,
            events=selected_events,
            evidence=evidence,
            outcome=outcome.value if outcome is not None else None,
        )

    @staticmethod
    def _sample_in_cluster(sample: ImprovingSample, cluster: Any) -> bool:
        run_ids = getattr(cluster, "run_ids", None)
        if run_ids is None:
            run_ids = getattr(cluster, "sample_ids", None)
        if run_ids is None:
            return True
        return sample.run.id in set(run_ids)


def _extract_user_feedback(
    messages: Sequence[Message],
    max_chars: int,
) -> tuple[str, ...]:
    """Keep bounded user messages; exclude system/tool noise."""

    values: list[str] = []
    used = 0
    for message in reversed(messages):
        if message.role is not MessageRole.USER or not message.content:
            continue
        text = " ".join(message.content.split()).strip()
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        values.append(text)
        used += len(text)
    values.reverse()
    return tuple(values)


def _render_events(events: Sequence[AgentEvent]) -> str:
    """Small fallback renderer used before TraceEvidenceBuilder is wired in."""

    lines: list[str] = []
    for event in events:
        parts = [event.type.value]
        if event.tool_call is not None:
            parts.append(event.tool_call.name)
        if event.tool_result is not None:
            parts.append("success" if event.tool_result.success else "failed")
        if event.error is not None:
            parts.append(str(event.error))
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )


__all__ = [
    "ImprovingSample",
    "PatternMiner",
    "ProcedureDistiller",
    "RunImprovingWatermark",
    "SkillImproveResult",
    "SkillImproveService",
]
