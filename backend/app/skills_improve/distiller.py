"""DISTILL：把聚类提炼成可复用 Skill Candidate。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING, Any

from app.model.config import Message, MessageRole, ModelRequest
from app.memory.utils import strip_json

from .models import DistillationResult, SkillCandidate, TaskCluster
from .service import ImprovingSample

if TYPE_CHECKING:
    from app.model.adapter import ModelAdapter

logger = logging.getLogger("sidekick.skills_improve.distiller")

# _SUCCESS_ANALYST_PROMPT = """你是成功轨迹分析师。
# 只分析首轮 outcome=passed 的 Run，目标是从多个成功案例中提炼少量稳定、可复用的做法，而不是复述某一道题的解法或设计固定流程。

# 请重点回答：
# 1. 哪些操作顺序、检查动作或验证方式在多个 Run 中重复出现？
# 2. 哪些接口、文件结构或约束被正确保留，并且对成功有明确作用？
# 3. 哪些做法可以迁移到相同类型的未来任务？明确适用条件；不适用时不要推荐。
# 4. 哪些验证信号能够说明步骤确实有效？

# 只使用输入中的首轮 user_request、trace_summary 和 evidence。不要使用 feedback、隐藏测试或工作区外信息。
# 只有至少两个独立 Run 体现相同做法时才提炼；单题名称、具体命令、函数名、输入值和偶然的工具顺序不要泛化。
# 输出结构化中文要点，包含“可复用做法、适用条件、验证信号”；如果没有稳定模式，明确写“无稳定模式”，不要为了凑结论而提出建议。
# """
_SUCCESS_ANALYST_PROMPT = """
Role:
You are an expert in success-pattern analysis for AI agent systems.

Mission:
Given one or more successful agent trajectories, identify behavior patterns that
are plausibly connected to producing the correct result and that can transfer to
similar tasks.

Requirements:

1. Evidence-based attribution
   Separate actions that likely contributed to correctness from incidental actions.
   Do not treat every action before success as a useful strategy.

2. Broad but selective coverage
   Cover the important reusable behaviors in the trajectories, but omit repeated
   low-level actions, task-specific constants, and irrelevant tool calls.

3. Frequency awareness
   When analyzing multiple successful trajectories, prioritize patterns that recur
   across different tasks. Repetition inside one trajectory is not independent
   evidence.

4. Generalization
   Describe the underlying mechanism, decision rule, implementation pattern,
   API discipline, boundary handling, or verification habit. Do not simply restate
   a single task's solution.

5. Concrete evidence
   Every pattern must include short examples of the observed behavior and explain
   why it plausibly helped.

6. Avoid overclaiming
   If a behavior appears useful but its causal role is uncertain, mark it as
   uncertain instead of presenting it as a required rule.

Output JSON only:

{
  "success_memory_items": [
    {
      "title": "...",
      "pattern": "...",
      "why_it_helps": "...",
      "examples": ["...", "..."],
      "scope": "...",
      "confidence": 0.0
    }
  ]
}

Rules:

- Return only the most useful reusable patterns.
- Do not output generic advice such as “test more” or “read carefully”.
- Do not include task answers, private file names, or one-off constants.
- Prefer implementation and decision patterns over descriptions of tool-call order.
- Keep the result compact enough for later merging with failure memories.
"""

# _FAILURE_ANALYST_PROMPT = """你是故障分析师。
# 只分析首轮 outcome=failed 的 Run，目标是从多个失败案例中提炼少量可观察、可复用的失败模式和提前修复动作，而不是把失败过程照抄成操作步骤。

# 请重点回答：
# 1. 失败发生在哪个阶段，具体症状是什么？区分编译/接口、工具调用、逻辑、边界、环境和验证不足等类型。
# 2. 哪些错误或行为在多个 Run 中重复出现？证据是什么？
# 3. 最可能的根因是什么？必须区分证据支持的根因和无法确认的推测。
# 4. Agent 应该在什么时候做什么检查，才能提前发现或避免该失败？只给出与证据直接相关的动作。
# 5. 修复后应通过什么可观察信号验证，哪些做法不应写进推荐流程？

# 只使用输入中的首轮 user_request、trace_summary 和 evidence。不要使用 feedback、隐藏测试或工作区外信息，也不要建议查看或复制隐藏测试。
# 只有至少两个独立 Run 出现相同模式，或单个 Run 提供了明确且可验证的根因时才提炼；不要把某个题目的错误消息、具体文件名或环境偶然问题泛化。
# 输出结构化中文要点，包含“失败症状、证据、根因、预防/修复动作、验证信号”；如果根因无法确认，明确标注不确定，不要输出 Skill JSON，也不要把失败过程本身写成成功经验。
# """
_FAILURE_ANALYST_PROMPT = """
Role:
You are an expert failure-analysis agent for the given task domain.

Mission:
Given an agent's execution artifacts, including logs, tool calls, produced files,
test results, and—only in this offline analysis stage—the ground-truth solution,
diagnose why the agent failed.

Your analysis must be systematic, evidence-driven, and reproducible.
Do not guess when the cause can be verified.

The ground-truth solution is analysis-only information. Never include it in the
agent's prompt or recommend copying it directly. Preserve the original artifacts
and perform any validation in an isolated temporary copy.

Required workflow:

1. Identify the exact mismatch between the agent output and the required behavior.
2. Trace the mismatch to a concrete agent decision, implementation step,
   tool invocation, environment issue, or incorrect assumption.
3. State the causal chain:
   symptom → agent behavior → violated rule → observed failure.
4. Validate the suspected cause with the smallest possible change in an isolated
   copy, then re-run the relevant check against the specification, oracle,
   test result, or ground truth.
5. If the validation fails, revise the diagnosis instead of claiming success.
6. Separate implementation bugs, incorrect self-tests, API/format mistakes,
   environment/tooling failures, and incomplete work.

Output JSON only:

{
  "failure_cause_items": [
    {
      "symptom": "...",
      "evidence": "...",
      "cause": "...",
      "agent_behavior": "...",
      "validation": "...",
      "confidence": 0.0
    }
  ],
  "failure_memory_items": [
    {
      "trigger": "...",
      "action": "...",
      "avoid": "...",
      "validation": "..."
    }
  ]
}

Rules:

- Include no more than 3 failure memory items.
- Each memory item must be a general rule that can transfer to similar tasks.
- Do not merely restate the error message.
- Do not give generic advice such as “test more” or “check carefully”.
- Do not claim a root cause unless the artifacts or isolated validation support it.
- If the evidence is insufficient, mark the cause as uncertain.
"""

# _MULTI_TEACHER_MERGE_APPENDIX = """

# ## Multi-teacher 合并要求
# - 输入中的“成功分析师”和“故障分析师”都是根据首轮轨迹提取的证据摘要，必须同时阅读，不能只依据成功分析师生成 Skill。
# - 如果故障分析师指出了重复的失败症状和有证据支持的根因，最终 Skill 必须在 `When to use`、`Instructions` 或 `Verification` 中体现对应的提前检查、规避动作或修复动作。
# - 不要把失败报告原文整段复制进 Skill；将它转换成“出现什么信号时，检查什么并如何修复”的可执行规则。
# - 先建立一个候选错误表：`症状 → 证据支持的根因 → 预防检查/修复动作 → 验证信号`。只把与当前成功流程同一模式、且在多个 Run 中重复或根因明确的条目写入 Skill。
# - 如果 Skill 中确实存在可复用的失败模式，可以在正文中增加 `## Common failures`，每条用一两句话说明触发信号、原因和解决办法；不要罗列所有分析到的错误。
# - 只保留两位分析师都能支持、或某一方有明确重复证据支持的规则；不要因为模板完整而补充固定的 selftest、模块检查、边界类型或清理步骤。
# - 一个 Skill 只应包含少量核心规则；如果分析结果来自多个互不相关的任务，返回 `skip`，不要用宽泛的“先读取并测试”把它们强行合并。
# - 如果成功分析和故障分析属于不同任务、无法形成同一条流程，返回 `skip: true`，不要强行拼接成泛化 Skill。
# """
_MULTI_TEACHER_MERGE_APPENDIX = """
You are a skill edit coordinator.

You receive multiple independently proposed edits derived from successful and
failed agent trajectories. Merge them into one coherent, concise, and
generalizable update to the skill.

Guidelines:

1. Deduplicate
   Merge edits that express the same behavior, failure mode, or implementation
   rule. Keep the version that is most specific, evidence-based, and reusable.

2. Resolve conflicts
   When success and failure edits conflict, prefer the edit supported by the
   stronger observed evidence and the clearer task contract. A verified failure
   should usually become a constraint or warning. A success pattern should become
   a required procedure only when it recurs across multiple independent tasks.

3. Preserve unique insights
   Keep distinct implementation patterns, API constraints, boundary rules,
   environment precautions, and failure-prevention steps. Remove task-specific
   answers, private file names, and incidental tool-call details.

4. Generalize carefully
   Convert repeated observations into a general rule with an explicit scope.
   Do not treat repeated wording, repeated actions within one trajectory, or
   correlated patches as independent evidence.

5. Keep the result concise
   Prefer a small number of high-value rules. Do not add generic instructions
   such as “test more”, “check carefully”, or “handle all edge cases” without a
   concrete risk and a bounded action.

6. Preserve causal information
   When possible, retain the relationship:
   observable symptom → agent behavior → root cause → prevention or repair.

7. Keep edits independent
   No two edits may target the same rule or overlapping passage. If two edits
   affect the same section, merge them into one final edit.

8. Preserve file consistency
   If the output uses file operations, a file creation and the corresponding
   SKILL.md link must be kept or removed together.

9. Preserve the existing skill contract
   Do not remove valid existing instructions unless the input evidence shows that
   they are harmful, contradictory, or obsolete.

10. Output one merged result
    Produce one final skill update, not one skill per patch and not a list of
    unresolved alternatives.

Output JSON only, using exactly one of these top-level shapes:

{"skip": true, "skip_reason": "why the evidence cannot form one skill"}

or:

{"skip": false, "name": "skill-name", "description": "when this skill applies", "content": "the complete SKILL.md content"}

For a non-skipped result, `name`, `description`, and `content` are required
top-level string fields. Do not nest them under `skill`, `candidate`, or another
object, and do not return Markdown fences around the JSON.
"""

_SYSTEM_PROMPT = """# 角色与职责
你负责把一组「首轮盲测 Run」中反复出现、且有证据支持的解决策略，提炼成可执行的 Agent Skill。
Skill 会在未来任务开始前提供给 Agent。它应是一份短操作手册，而不是一次任务的复盘或“坑/做法”摘要；错误泛化比少生成一条 Skill 更糟。

## 生成前检查
- 同一条规则必须在至少两个独立样本的首轮证据中出现；只有一个样本、样本互相矛盾或只是偶然实现细节时，返回 skip。
- 只能使用首轮 evidence 和 trace_summary 中可观察的信息。不要使用、推测或补写 feedback、隐藏测试、canonical-data、缓存文件或工作区外的信息。
- `outcome=passed` 是成功轨迹：提炼可复用的操作顺序、检查方式和验证方法；`outcome=failed` 是失败轨迹：提炼失败症状、根因和应该提前采取的修复动作。不要把失败过程本身写成推荐做法。
- 先分别归纳两类轨迹，再做对照：成功轨迹说明“什么有效”，失败轨迹说明“什么应避免或如何修复”。只有一类轨迹时，不要虚构另一类证据。
- 一个 cluster 只能生成一个 Skill Candidate；当两个区块都有内容时，必须把成功步骤、失败症状和对应修复合并到同一份 Skill 中，不要按 outcome 生成多个 Skill。
- 不要把单题名称、常量、测试数量、完整错误字符串或某个实现的偶然写法泛化成跨语言规则。
- 不要把某次评测环境的限制写成通用规则，例如“只能使用某个语言/命令”“必须使用 heredoc”“不能查资料”；只有在多个样本中都能观察到且确实属于任务约束时才保留。
- 不要机械罗列空输入、负数、极值等边界；验证步骤必须根据题目实际输入域选择正常、边界、退化和错误分支。
- 不要要求删除 `__pycache__`、`.pyc` 或其他运行时缓存；可以清理本次创建的临时文件，但不得删除已有环境文件。
- 不要把一个具体示例中的函数名、模块名、输入值或输出值写成通用模板；示例必须服务于规则说明，且不能诱导 Agent 修改无关文件。
- 对跨文件任务不要绝对要求“只改一个文件”；应要求优先最小局部修改目标文件，并保留项目已有 API、模块结构和无关文件。
- 对 JavaScript/TypeScript 任务，如果轨迹显示模块类型、导出形式或加载方式造成过失败，可以提炼相应检查；不要无证据地要求读取特定配置文件。
- 如果轨迹显示模块加载本身是风险，验证中可以加入 `import` 或 `require` 检查；否则不必强行添加。
- 只有轨迹表明规则复杂且普通用例不足以发现分歧时，才建议独立参考实现或参数扫描。
- 不得查找、复制或猜测隐藏测试，也不得把缓存内容当作验证依据。
- 如果无法写出明确的触发信号、执行步骤和验证方式，返回 skip。

## Skill 文档写法
- `description` 用一句话说明“什么时候使用”和“要采取什么动作/得到什么结果”，例如“当运行时出现 X 时，先检查 Y，再按 Z 修复”。不要写成泛化的“实现 Exercism 题目”。
- `content` 是 Markdown，内容不超过 500 个中文字符。可以使用 `## When to use`、`## Instructions`、`## Verification` 和 `## Example` 组织内容，但只保留对该 Skill 有证据支持的部分，不要为了满足模板而补步骤。
- `When to use` 应说明可观察触发信号；`Instructions` 应写清必要的检查和动作；`Verification` 应给出可观察的验证方式；`Example` 只有在能提供准确、非偶然的例子时才添加。
- 如果失败分析中存在与主流程直接相关的可复用错误，可以增加 `## Common failures`；每条写清症状、原因和解决办法，数量保持少而具体。
- 步骤必须要求最小局部修改，保留已有 API、签名和已经通过的行为；禁止因为一个失败断言重写整个算法。
- 验证步骤应优先使用项目已有的解释器、测试入口或本地可观察行为；只有证据明确支持时，才指定具体命令或语言工具。
- 如果 Skill 包含边界用例，应依据题目实际输入域选择；不要机械要求空输入、负数、极值、非方阵或“非法语法”等不一定成立的输入。
- 如果 Skill 包含示例，示例必须自洽且与前文接口、输入输出格式一致；无法确认示例正确时省略示例，不要编造具体值。
- 不得建议查看或反汇编隐藏测试、`__pycache__`/`.pyc`、缓存或工作区外文件，也不得复制测试文件。

## 输出格式
只输出合法 JSON，不要 Markdown 围栏或解释：
{
  "skip": false,
  "skip_reason": null,
  "name": "lowercase-hyphenated-name",
  "description": "一句话说明触发场景和动作",
  "content": "## When to use\\n...\\n\\n## Instructions\\n1. ...\\n\\n## Verification\\n...\\n\\n## Example\\n..."
}
- 无法同时满足证据、同类任务、明确步骤和可验证结果时，返回 `skip: true` 并说明原因。
- 若 `skip` 为 true，`name`、`description`、`content` 可以为 null；否则三者都必须非空。
"""


class ModelProcedureDistiller:
    """用主模型把 TaskCluster 蒸馏成 SkillCandidate。"""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        model: str | None,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4_000,
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
    ) -> None:
        self._model_adapter = model_adapter
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts

    async def distill(
        self,
        cluster: Any,
        *,
        samples: tuple[ImprovingSample, ...],
        catalog: Sequence[Any],
        skill_loader: Callable[[str], Awaitable[Any | None]],
    ) -> DistillationResult:
        payload = self._render_prompt(cluster, samples, catalog)
        model = self._model or self._model_adapter.default_model
        async with asyncio.timeout(self._timeout_seconds):
            for _attempt in range(1, self._max_attempts + 1):
                request = ModelRequest(
                    messages=(
                        Message(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
                        Message(role=MessageRole.USER, content=payload),
                    ),
                    model=model,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    extra_body=_structured_output_options(self._model_adapter),
                )
                response = await self._model_adapter.complete(request)
                output = response.message.content or ""
                try:
                    data = _load_json_object(output)
                    if data.get("skip"):
                        return DistillationResult(
                            skip_reason=str(data.get("skip_reason") or "skipped")
                        )
                    candidate = SkillCandidate(
                        name=str(data["name"]),
                        description=str(data["description"]),
                        content=str(data["content"]).strip(),
                        source_run_ids=tuple(s.run.id for s in samples),
                        cluster_id=getattr(cluster, "id", None),
                    )
                    return DistillationResult(candidate=candidate)
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning(
                        "distiller invalid output (attempt %s): %s", _attempt, exc
                    )
        raise ValueError("procedure distiller returned invalid output")

    @staticmethod
    def _render_prompt(
        cluster: Any,
        samples: tuple[ImprovingSample, ...],
        catalog: Sequence[Any],
    ) -> str:
        # The shared renderer is defined below with the multi-teacher code;
        # keep it available to the original cluster distiller as well.
        return ModelMultiTeacherDistiller._render_prompt(cluster, samples, catalog)


class ModelMultiTeacherDistiller(ModelProcedureDistiller):
    """分别分析成功/失败 Run，再合并为一个 Skill。"""

    def __init__(self, *args: Any, analysis_dir: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._analysis_dir = Path(analysis_dir) if analysis_dir is not None else None

    async def distill_multi_teacher(
        self,
        *,
        samples: tuple[ImprovingSample, ...],
        catalog: Sequence[Any],
    ) -> DistillationResult:
        passed = tuple(sample for sample in samples if sample.outcome == "passed")
        failed = tuple(sample for sample in samples if sample.outcome == "failed")
        positive = await self._analyze("success", _SUCCESS_ANALYST_PROMPT, passed)
        negative = await self._analyze("failure", _FAILURE_ANALYST_PROMPT, failed)
        payload = (
            "请把下面两位分析师的结果合并成一个可复用 Skill。"
            "只有证据支持的规则才能保留；不同任务无法合并时返回 skip。\n\n"
            "## 成功分析师\n" + positive + "\n\n"
            "## 故障分析师\n" + negative + "\n\n"
            "## 已有 Skill（用于去重）\n"
            + "\n".join(
                f"[{getattr(item, 'name', '')}] {getattr(item, 'description', '')}"
                for item in catalog
                if getattr(item, "name", "")
            )
        )
        if failed:
            payload += (
                "\n\n本批存在失败轨迹，因此输出的 content 必须包含 `## Common failures`。"
                "至少列出一个与成功流程相关、且有证据支持的条目，格式为："
                "症状；根因；预防/修复动作；验证信号。"
                "如果失败分析无法与成功流程合并，返回 skip，不要省略该章节。"
            )
        return await self._distill_payload(
            payload, tuple(samples), require_common_failures=bool(failed)
        )

    async def _analyze(
        self,
        label: str,
        analyst_prompt: str,
        samples: tuple[ImprovingSample, ...],
    ) -> str:
        if not samples:
            return "（无该类轨迹；不要推测。）"
        blocks = "\n".join(
            json.dumps(
                {
                    "run_id": sample.run.id,
                    "user_request": sample.user_request[:2_000],
                    "trace_summary": _trace_summary(sample),
                    "evidence": (sample.evidence or "")[:6_000],
                },
                ensure_ascii=False,
            )
            for sample in samples
        )
        request = ModelRequest(
            messages=(
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        analyst_prompt
                    ),
                ),
                Message(role=MessageRole.USER, content=blocks),
            ),
            model=self._model or self._model_adapter.default_model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        response = await self._model_adapter.complete(request)
        analysis = (response.message.content or "（分析为空。）")[:8_000]
        if self._analysis_dir is not None:
            await asyncio.to_thread(
                self._save_analysis,
                label,
                samples,
                analysis,
            )
        return analysis

    def _save_analysis(
        self,
        label: str,
        samples: tuple[ImprovingSample, ...],
        analysis: str,
    ) -> None:
        assert self._analysis_dir is not None
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "analysis_type": label,
            "created_at": time.time(),
            "run_ids": [sample.run.id for sample in samples],
            "analysis": analysis,
        }
        path = self._analysis_dir / f"{label}-{int(time.time())}-{uuid4().hex[:8]}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _distill_payload(
        self,
        payload: str,
        samples: tuple[ImprovingSample, ...],
        *,
        require_common_failures: bool = False,
    ) -> DistillationResult:
        request = ModelRequest(
            messages=(
                Message(
                    role=MessageRole.SYSTEM,
                    content= _MULTI_TEACHER_MERGE_APPENDIX,
                ),
                Message(role=MessageRole.USER, content=payload),
            ),
            model=self._model or self._model_adapter.default_model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            extra_body=_structured_output_options(self._model_adapter),
        )
        response = await self._model_adapter.complete(request)
        data = _load_json_object(response.message.content or "")
        if data.get("skip"):
            return DistillationResult(skip_reason=str(data.get("skip_reason") or "skipped"))
        content = str(data["content"]).strip()
        if require_common_failures and "common failures" not in content.lower():
            raise ValueError(
                "multi-teacher output omitted required ## Common failures section"
            )
        return DistillationResult(
            candidate=SkillCandidate(
                name=str(data["name"]),
                description=str(data["description"]),
                content=content,
                source_run_ids=tuple(sample.run.id for sample in samples),
                cluster_id="multi-teacher",
            )
        )

    @staticmethod
    def _render_prompt(
        cluster: Any,
        samples: tuple[ImprovingSample, ...],
        catalog: Sequence[Any],
    ) -> str:
        cluster_block = json.dumps(
            {
                "id": getattr(cluster, "id", ""),
                "title": getattr(cluster, "title", ""),
                "description": getattr(cluster, "description", ""),
            },
            ensure_ascii=False,
        )
        def render_sample(sample: ImprovingSample) -> str:
            return json.dumps(
                {
                    "run_id": sample.run.id,
                    "outcome": sample.outcome,
                    "user_request": sample.user_request[:2_000],
                    "trace_summary": _trace_summary(sample),
                    "evidence": (sample.evidence or "")[:6_000],
                },
                ensure_ascii=False,
            )

        passed_blocks = [
            render_sample(sample) for sample in samples if sample.outcome == "passed"
        ]
        failed_blocks = [
            render_sample(sample) for sample in samples if sample.outcome == "failed"
        ]
        unlabeled_blocks = [
            render_sample(sample)
            for sample in samples
            if sample.outcome not in {"passed", "failed"}
        ]
        catalog_lines = [
            f"[{getattr(item, 'name', '')}] {getattr(item, 'description', '')}"
            for item in catalog
            if getattr(item, "name", "")
        ]
        return "\n\n".join(
            [
                "下面的 Run 都是首轮盲测证据；只根据这些证据判断，不能把后续 feedback 当作经验。",
                "请先确认它们是否是同语言、同类问题且至少两个样本重复出现，再决定是否生成 Skill。成功和失败必须分别判断；没有标签的 Run 不能当作成功或失败证据。",
                "## 模式（cluster）\n" + cluster_block,
                "## Successful traces（首轮通过）\n"
                + ("\n".join(passed_blocks) if passed_blocks else "（无）"),
                "## Failed traces（首轮未通过）\n"
                + ("\n".join(failed_blocks) if failed_blocks else "（无）"),
                "## Unlabelled traces（不可用于判断结果）\n"
                + ("\n".join(unlabeled_blocks) if unlabeled_blocks else "（无）"),
                "## 已有 Skill 目录（用于去重/更新判断）\n"
                + ("\n".join(catalog_lines) if catalog_lines else "（空）"),
            ]
        )


def _load_json_object(content: str) -> dict[str, Any]:
    """Parse the first JSON object and tolerate trailing model commentary."""

    text = strip_json(content).lstrip()
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _end = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("distiller output must be a JSON object")
    return value


def _trace_summary(sample: ImprovingSample) -> dict[str, object]:
    """Render bounded, observable trace facts for distillation."""

    events = sample.events
    commands: list[str] = []
    paths: list[str] = []
    tool_errors: list[str] = []
    command_outputs: list[str] = []
    for event in events:
        call = event.tool_call
        result = event.tool_result
        if call is not None and isinstance(call.arguments, dict):
            command = call.arguments.get("command")
            if isinstance(command, str) and command.strip():
                commands.append(command.strip()[:240])
            for key in ("path", "file_path", "filename"):
                value = call.arguments.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip()[:200])
        if result is None:
            continue
        if result.output is not None and call is not None and call.name == "run_shell_command":
            command_outputs.append(str(result.output).replace("\n", " ")[:600])
        if not result.success:
            detail = result.error or ""
            if result.output is not None:
                detail = f"{detail} output={result.output}".strip()
            if detail:
                tool_errors.append(f"{result.tool_name}: {detail[:500]}")
    return {
        "run_status": sample.run.status.value,
        "stop_reason": sample.run.stop_reason,
        "tool_counts": dict(
            Counter(event.tool_call.name for event in events if event.tool_call is not None)
        ),
        "event_types": dict(Counter(event.type.value for event in events)),
        "commands": commands[-12:],
        "paths": paths[-30:],
        "tool_errors": tool_errors[-12:],
        "command_outputs": command_outputs[-12:],
        "skill_names": sorted({event.skill_name for event in events if event.skill_name}),
    }


def _structured_output_options(adapter: ModelAdapter) -> dict[str, object]:
    """Avoid spending the structured-output budget on DeepSeek reasoning."""

    if getattr(adapter, "provider", "") != "deepseek":
        return {}
    return {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


__all__ = ["ModelProcedureDistiller", "ModelMultiTeacherDistiller"]
