"""对已有评测 Run 跑一次 Skill Improve。"""

from __future__ import annotations

import asyncio
import argparse
import logging
import tempfile
from pathlib import Path

from app.application import Application  # noqa: F401  触发循环 import 重建
from app.model.config import rebuild_models

rebuild_models()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.conversation.store import ConversationStore
from app.model.registry import ModelAdapterRegistry
from app.run.config import RunStatus
from app.run.outcome_store import RunOutcomeStore
from app.run.store import RunStore
from app.skills.store import SkillStore
from app.skills_improve.candidate_store import SkillCandidateStore
from app.skills_improve.config import SkillImprovingSettings
from app.skills_improve.distiller import ModelMultiTeacherDistiller, ModelProcedureDistiller
from app.skills_improve.evidence import DefaultEventSelector, TraceEvidenceBuilder
from app.skills_improve.miner import ModelPatternMiner
from app.skills_improve.service import SkillImproveService
from app.trace.store import TraceStore

async def main(database: Path, data_dir: Path, improve_method: str = "cluster") -> None:
    settings = SkillImprovingSettings(
        enabled=True,
        improve_method=improve_method,
        provider="deepseek",
        batch_size=50,
        max_runs_per_scan=100,
        auto_accept=True,
        data_dir=data_dir,
    )

    registry = ModelAdapterRegistry()
    adapter = registry.get("deepseek")

    run_store = RunStore(database)
    trace_store = TraceStore(database)
    outcome_store = RunOutcomeStore(database)
    conversation_store = ConversationStore(database)
    skill_store = SkillStore()
    await run_store.initialize()
    await trace_store.initialize()
    await outcome_store.initialize()
    await conversation_store.initialize()
    await skill_store.initialize()

    completed = await run_store.list_runs(status=RunStatus.COMPLETED, limit=1000)
    print(f"completed runs 总数: {len(completed)}")

    candidate_store = SkillCandidateStore(
        data_dir / "candidates.db",
        skill_project_dir=skill_store.project_dir,
    )
    await candidate_store.initialize()

    miner = ModelPatternMiner(
        adapter, None,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
        max_attempts=settings.max_attempts,
    )
    distiller_class = (
        ModelMultiTeacherDistiller
        if settings.improve_method == "multi_teacher"
        else ModelProcedureDistiller
    )
    distiller = distiller_class(
        adapter, None,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
        max_attempts=settings.max_attempts,
        **({"analysis_dir": data_dir / "teacher-analyses"}
           if settings.improve_method == "multi_teacher" else {}),
    )

    service = SkillImproveService(
        run_store=run_store,
        trace_store=trace_store,
        skill_store=skill_store,
        candidate_store=candidate_store,
        registry=registry,
        settings=settings,
        conversation_store=conversation_store,
        outcome_store=outcome_store,
        miner=miner,
        distiller=distiller,
        selector=DefaultEventSelector(),
        evidence_builder=TraceEvidenceBuilder(),
    )

    outcome = await service.maybe_run_improving()
    print("\n=== outcome ===")
    print("triggered:", outcome.triggered)
    print("skipped:", outcome.skipped_reason)
    print("scanned:", outcome.scanned_run_count)
    print("samples:", outcome.sample_count)
    print("clusters:", outcome.cluster_count)
    print("candidates:", outcome.candidate_count)
    print("processed_run_ids:", outcome.processed_run_ids)
    print("errors:", outcome.errors)

    pending = await candidate_store.list_pending()
    print(f"\n=== 待审核候选（{len(pending)} 个）===")
    for c in pending:
        print(f"\n[{c.name}] {c.description}")
        print(f"  cluster={c.cluster_id} source_runs={c.source_run_ids}")
        print(f"  content:\n{c.content}")

    await registry.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对指定评测数据库运行一次 Skill Improve")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".database/polyglot-javascript.db"),
        help="评测数据库路径",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".skills/improving"),
        help="watermark 和候选 Skill 数据目录",
    )
    parser.add_argument(
        "--improve-method",
        choices=("cluster", "multi_teacher"),
        default="cluster",
        help="cluster：先聚类；multi_teacher：成功/失败分析后合并为一个 Skill",
    )
    args = parser.parse_args()
    asyncio.run(main(args.database.resolve(), args.data_dir.resolve(), args.improve_method))
