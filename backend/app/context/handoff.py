"""会话交接（handoff）数据模型。

当增量压缩已经无法放下当前会话历史时，AgentLoop 触发 handoff：
扩大摘要范围后继续执行，原始历史保持完整。HandoffSnapshot 保存摘要及 git
现场用于诊断，下一次 Run 的摘要仍由 ConversationSummaryStore 加载。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.context.summary import RollingConversationSummary


class GitSnapshot(BaseModel):
    """纯读的 git 情报，handoff 时采集，下一个 agent 用来理解工作现场。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    head_sha: str | None = None
    branch: str | None = None
    changed_files: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()

    @field_validator(
        "changed_files", "untracked_files", mode="before"
    )
    @classmethod
    def _normalize_paths(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        values = (value,) if isinstance(value, str) else value
        return tuple(str(v).strip() for v in values if str(v).strip())

    def render_markdown(self) -> str:
        lines = []
        if self.head_sha or self.branch:
            git_lines = ["```"]
            if self.branch:
                git_lines.append(f"branch: {self.branch}")
            if self.head_sha:
                git_lines.append(f"HEAD:   {self.head_sha[:12]}")
            git_lines.append("```")
            lines.append("### Git 状态")
            lines.extend(git_lines)
            lines.append("")
        if self.changed_files:
            lines.append("### 已修改文件")
            lines.extend(f"- `{f}`" for f in self.changed_files)
            lines.append("")
        if self.untracked_files:
            lines.append("### 未跟踪文件")
            lines.extend(f"- `{f}`" for f in self.untracked_files)
            lines.append("")
        return "\n".join(lines)


class HandoffSnapshot(RollingConversationSummary):
    """Reset 时写入、交接给下一个 agent 的完整状态快照。

    继承 RollingConversationSummary 的所有 PI 字段，再加上 handoff 特有的
    元数据（git 快照、handoff 原因、消息覆盖计数等）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    git_state: GitSnapshot | None = None
    summary_covered_message_count: int = 0
    handoff_reason: str | None = None

    step_at_handoff: int | None = None
    model_calls_at_handoff: int | None = None

    def render_as_system_message(self) -> str:
        """渲染为下一个 agent 的初始 system message 内容。"""

        lines = [
            "## 🔄 会话已交接（Context Handoff）",
            "",
            f"上一个 agent 在处理了 {self.summary_covered_message_count} 条消息后因 {self.handoff_reason or '上下文溢出'} 触发 handoff。",
            "以下是完整的工作状态快照，据此继续完成任务。",
            "",
        ]

        if self.step_at_handoff is not None or self.model_calls_at_handoff is not None:
            budget_hint_parts = []
            if self.step_at_handoff is not None:
                budget_hint_parts.append(f"已执行 {self.step_at_handoff} 步")
            if self.model_calls_at_handoff is not None:
                budget_hint_parts.append(f"已调用模型 {self.model_calls_at_handoff} 次")
            budget_hint = "、".join(budget_hint_parts)
            lines.append(f"> ⚠️ {budget_hint}，请关注剩余预算，必要时尽快收尾交付。")
            lines.append("")

        lines.append(self.render_markdown())

        if self.git_state is not None:
            lines.append("---")
            lines.append("")
            lines.append("## 工作现场快照")
            lines.append("")
            lines.append(self.git_state.render_markdown())

        return "\n".join(lines)


async def collect_git_snapshot(cwd: str | Path | None = None) -> GitSnapshot | None:
    """采集当前工作目录的 git 情报。纯读操作，不做任何写。

    返回 None 表示不是 git 仓库或 git 命令执行失败——这是正常情况，
    handoff 不应该因为 git 采集失败就阻塞。
    """

    resolved_cwd = str(cwd) if cwd is not None else os.getcwd()

    def _run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=resolved_cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    branch = await asyncio.to_thread(_run, ["rev-parse", "--abbrev-ref", "HEAD"])
    head_sha = await asyncio.to_thread(_run, ["rev-parse", "HEAD"])
    changed_raw = await asyncio.to_thread(_run, ["diff", "--name-only", "HEAD"])
    untracked_raw = await asyncio.to_thread(_run, ["ls-files", "--others", "--exclude-standard"])

    if branch is None and head_sha is None:
        return None

    return GitSnapshot(
        head_sha=head_sha,
        branch=branch,
        changed_files=tuple(
            line for line in (changed_raw or "").splitlines() if line
        ),
        untracked_files=tuple(
            line for line in (untracked_raw or "").splitlines() if line
        ),
    )
