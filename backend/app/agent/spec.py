"""Agent 规格加载：从 agent.md 文件读取 system prompt。

查找优先级（从高到低）：
1. 显式传入的 path
2. <workspace>/agent.md （项目级）
3. ~/.sidekick/agent.md  （用户级）
4. 内置默认文本          （兜底）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_BUILTIN_FALLBACK = (
    "你是一个本地运行的智能助理。请使用用户的语言回答。"
    "调用工具时优先使用已有结果；网页搜索通常只需一到两次，获得可用结果后"
    "立即整理回答，不要为了追求完美而反复改写相同查询。"
    "只有用户目标确实依赖实时或外部信息、或者需要操作本地环境时才调用工具；"
    "普通知识问答、能力说明，以及请求的工具不存在时直接如实回答，不要为了"
    "试探或确认而调用搜索、文件或其他无关工具。"
    "当用户明确要求记录任务，或工作复杂、需要多个步骤或跨多轮跟踪时，"
    "调用 task_create。一个用户整体目标通常只创建一个 Task，目标内部的阶段、"
    "模块和动作应拆为该 Task 的 Steps；只有彼此独立、可分别完成和关闭的目标"
    "才创建多个 Task。简单的一次性问题不要创建任务。完成任务步骤、计划"
    "变化或任务状态变化后调用 task_update，必要时用 task_get/task_list"
    "重新确认任务状态。"
    "模型当前看到的是受预算控制的工作上下文，不等于原始事实已被删除。"
    "当摘要缺少旧对话中的用户约束或决定时，用 tool_search 激活 history_search/"
    "history_read；当历史工具输出被截断、清理或摘要时，激活 evidence_search/"
    "evidence_read 按需取回不可变原文。无法取回时如实说明，不要凭摘要补造。"
    "如果生成了用户需要保留、下载或查看的文件（如报告、CSV、代码、图片），"
    "在最终回答前调用 artifact_publish 发布它；如果最终交付的是结果链接，"
    "也用 artifact_publish 发布。普通中间文件、临时文件、Trace、"
    "Computer Screenshot 不要发布为 Artifact；没有实际交付物时不要调用"
    "artifact_publish。"
)

_USER_AGENT_MD_PATH = Path.home() / ".sidekick" / "agent.md"


def _read_agent_md(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not text:
        return None
    return text


def load_agent_prompt(
    explicit_path: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
) -> str:
    """按优先级加载 agent.md，返回可直接用作 system prompt 的纯文本。

    Args:
        explicit_path: 调用方显式指定的 agent.md 路径（最高优先级）。
        workspace_root: 当前工作区根目录；会查找 <workspace_root>/agent.md。

    Returns:
        agent.md 的文本内容；所有候选都不存在时返回内置 fallback。
    """
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser().resolve())

    if workspace_root is not None:
        candidates.append(Path(workspace_root).expanduser().resolve() / "agent.md")

    candidates.append(_USER_AGENT_MD_PATH)

    for candidate in candidates:
        text = _read_agent_md(candidate)
        if text is not None:
            return text

    return _BUILTIN_FALLBACK


def resolve_agent_md_path(
    explicit_path: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
) -> Path | None:
    """返回实际被采用的 agent.md 路径；全部 fallback 时返回 None（表示用内置默认）。"""
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser().resolve())

    if workspace_root is not None:
        candidates.append(Path(workspace_root).expanduser().resolve() / "agent.md")

    candidates.append(_USER_AGENT_MD_PATH)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None
