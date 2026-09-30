"""Sidekick CLI 展示与模型设置：直接使用 ModelConfig 和 .config 文件。"""
from __future__ import annotations

import asyncio
import getpass
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.model.config import ModelConfig, ModelProvider
from dotenv import set_key
from pydantic import SecretStr

if TYPE_CHECKING:
    from app.agent.events import AgentEvent
    from app.checkpoint.config import RunCheckpoint
    from app.memory.regular import RegularMemoryRecord
    from app.run.config import Run
    from app.tools.permissions.models import PermissionRule


_PROVIDER_LABELS = {
    ModelProvider.OPENAI: "OpenAI",
    ModelProvider.QWEN: "Qwen",
    ModelProvider.DEEPSEEK: "DeepSeek",
}


def print_banner(*, output_fn: Callable[[str], Any] = print) -> None:
    """Print a compact, fixed-width Claude-style welcome panel."""
    use_color = output_fn is print and not os.environ.get("NO_COLOR")

    def style(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    left_width = 28
    right_width = 40
    inner_width = 75

    def cell(text: str, width: int, code: str) -> str:
        # Pad before adding ANSI escapes so terminal alignment stays exact.
        return style(text[:width].ljust(width), code)

    left = [
        ("Welcome back!", "1;38;5;255"),
        ("", "38;5;245"),
        ("          .-\"\"-.       ", "38;5;33"),
        ("       .-'  .--. '-.    ", "38;5;33"),
        ("     .'   .'    '.   '.  ", "38;5;33"),
        ("    /    /  @  @  \\    \\ ", "38;5;33"),
        ("   ;    |    ^     |    ;", "38;5;33"),
        ("   |     \\  ---  /     |", "38;5;33"),
        ("    \\     '.___.'     / ", "38;5;33"),
        ("     '._           _.'  ", "38;5;33"),
        ("        '---.___.---'   ", "38;5;33"),
        ("", "38;5;245"),
        ("Memory-aware assistant", "38;5;248"),
        ("    + tools + reflection", "38;5;245"),
    ]
    right = [
        ("Tips for getting started", "1;38;5;216"),
        ("/help      list commands", "38;5;117"),
        ("/new       start a fresh conversation", "38;5;117"),
        ("/memories  inspect long-term memory", "38;5;117"),
        ("/trace     inspect a run and reflection", "38;5;117"),
        ("", "38;5;245"),
        ("Type a message to begin.", "3;38;5;245"),
    ]

    lines = [
        "",
        style(
            "╭"
            + ("─ Sidekick CLI · local agent workspace"[: inner_width - 2]).ljust(
                inner_width - 2, "─"
            )
            + "╮",
            "1;38;5;216",
        ),
    ]
    for index in range(max(len(left), len(right))):
        left_text, left_code = left[index] if index < len(left) else ("", "38;5;245")
        right_text, right_code = right[index] if index < len(right) else ("", "38;5;245")
        lines.append(
            "│ "
            + cell(left_text, left_width, left_code)
            + " │ "
            + cell(right_text, right_width, right_code)
            + " │"
        )
    lines.extend([style("╰" + "─" * (inner_width - 2) + "╯", "38;5;238"), ""])
    output_fn("\n".join(lines))


def print_startup_status(
    rows: Sequence[tuple[str, str]],
    *,
    notices: Sequence[str] = (),
) -> None:
    use_color = (
        getattr(sys.stdout, "isatty", lambda: False)()
        and not os.environ.get("NO_COLOR")
    )
    label_colors = {
        "主模型": "38;5;117",      # blue
        "会话": "38;5;183",        # purple
        "搜索": "38;5;114",        # green
        "长期记忆": "38;5;221",    # yellow
    }

    def style(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    print()
    for label, value in rows:
        label_text = style(f"  {label}：", label_colors.get(label, "38;5;250"))
        print(f"{label_text}{style(value, '38;5;252')}")
    for notice in notices:
        print(f"{style('  提醒：', '38;5;214')}{notice}")
    print(
        f"\n{style('  输入任务开始工作 · /help 查看命令 · /new 新建会话 · /exit 退出', '38;5;245')}"
    )


def print_conversation_divider(label: str = "会话") -> None:
    """Print a visible boundary between conversations or input turns."""
    columns = shutil.get_terminal_size((80, 24)).columns
    width = max(48, min(columns, 120))
    marker = f" {label} "
    remaining = max(0, width - len(marker) - 2)
    left = remaining // 2
    right = remaining - left
    print(f"\n{'─' * left}{marker}{'─' * right}")


async def run_setup(
    *,
    settings: ModelConfig | None = None,
    config_path: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], Any] = print,
) -> bool:
    path = Path(config_path or ModelConfig.model_config["env_file"]).expanduser().absolute()
    try:
        current = settings if settings is not None else ModelConfig(_env_file=path)
        print_banner(output_fn=output_fn)
        output_fn(f"配置保存到：{path}")
        output_fn("API Key 将保存在该配置文件中，文件权限设为仅当前用户读写。")
        output_fn("进程环境变量优先于文件；已有环境变量覆盖时，需同步调整环境变量。")
        selected = await _choose_provider(current, input_fn, output_fn)
        prefix = selected.value
        old_model = getattr(current, f"{prefix}_model")
        old_url = getattr(current, f"{prefix}_base_url")
        old_key = getattr(current, f"{prefix}_api_key")

        model = (await asyncio.to_thread(input_fn, f"模型名称 [{old_model}]：")).strip() or old_model
        url_input = (await asyncio.to_thread(
            input_fn, f"API 地址 [{old_url or '默认地址'}]（回车保留）："
        )).strip()
        base_url = url_input or old_url
        has_key = old_key is not None and bool(old_key.get_secret_value().strip())
        hint = "回车保留现有密钥" if has_key else "必填，输入不回显"
        new_key = (await asyncio.to_thread(secret_fn, f"API Key（{hint}）：")).strip()
        if not new_key and not has_key:
            output_fn("未提供 API Key，设置已取消。")
            return False

        # 用新值构造配置，不修改调用方传入的 settings 对象。
        values = current.model_dump()
        values.update({
            "model_default_provider": selected,
            f"{prefix}_model": model,
            f"{prefix}_base_url": base_url,
        })
        if new_key:
            values[f"{prefix}_api_key"] = SecretStr(new_key)
        candidate = ModelConfig.model_validate(values)
        candidate.load_provider_config(selected)

        output_fn(f"\n将保存：{_PROVIDER_LABELS[selected]} / {model}")
        output_fn(f"API 地址：{base_url or '默认地址'}")
        if not await _confirm("确认保存？[Y/n] ", input_fn):
            output_fn("设置已取消。")
            return False
        updates = {
            "MODEL_DEFAULT_PROVIDER": selected.value,
            f"{prefix.upper()}_MODEL": model,
            f"{prefix.upper()}_BASE_URL": base_url or "",
        }
        # 回车保留密钥时，不把环境变量中的密钥额外复制到文件中。
        if new_key:
            updates[f"{prefix.upper()}_API_KEY"] = new_key
        try:
            await asyncio.to_thread(_save_config, path, updates)
        except Exception as exc:
            output_fn(f"保存失败（{type(exc).__name__}），原配置未被替换。")
            return False
        output_fn("配置已保存。")

        if await _confirm("立即测试连接？[Y/n] ", input_fn):
            output_fn("正在测试模型连接……")
            try:
                elapsed = await _test_connection(candidate, selected)
            except Exception as exc:
                # Provider SDK 的异常正文不包含 API Key，保留它才能区分
                # 模型不存在、认证失败、请求路径错误和网络错误。
                output_fn(
                    f"连接测试失败（{type(exc).__name__}）：{exc}\n"
                    "已保存的配置保留。"
                )
            else:
                output_fn(f"连接成功：{selected.value}/{model} · {elapsed:.0f}ms")
        return await _confirm("现在进入 Sidekick？[Y/n] ", input_fn)
    except (EOFError, KeyboardInterrupt):
        output_fn("\n设置已取消。")
        return False


async def _choose_provider(
    settings: ModelConfig,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> ModelProvider:
    providers = tuple(_PROVIDER_LABELS)
    default = settings.model_default_provider
    if default not in providers:
        default = providers[0]
    default_index = providers.index(default) + 1
    output_fn("选择主模型提供方：")
    for index, provider in enumerate(providers, 1):
        key = getattr(settings, f"{provider.value}_api_key")
        configured = key is not None and bool(key.get_secret_value().strip())
        marker = "（已配置）" if configured else ""
        output_fn(f"  {index}. {_PROVIDER_LABELS[provider]} {marker}")
    while True:
        value = (await asyncio.to_thread(input_fn, f"请选择 [{default_index}]：")).strip().lower()
        if not value:
            return default
        if value.isdigit() and 1 <= int(value) <= len(providers):
            return providers[int(value) - 1]
        for provider in providers:
            if value == provider.value:
                return provider
        output_fn("请输入列表编号或提供方名称。")


async def _confirm(prompt: str, input_fn: Callable[[str], str]) -> bool:
    while True:
        answer = (await asyncio.to_thread(input_fn, prompt)).strip().lower()
        if answer in {"", "y", "yes", "是"}:
            return True
        if answer in {"n", "no", "否"}:
            return False
        prompt = "请输入 y 或 n："


def _save_config(path: Path, updates: dict[str, str]) -> None:
    """在临时文件更新选中字段，再原子替换；保留其他设置和注释。"""
    if path.is_symlink():
        raise ValueError("config file cannot be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        for key, value in updates.items():
            set_key(str(temporary), key, value, quote_mode="always", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def _test_connection(settings: ModelConfig, provider: ModelProvider) -> float:
    # 仅选择测试时加载模型依赖，配置界面不依赖 Registry 或设置服务。
    from app.model.config import Message, MessageRole
    from app.model.adapter import ModelCompatibleAdapter
    from app.model.config import ModelRequest

    adapter = ModelCompatibleAdapter(settings.load_provider_config(provider))
    started = perf_counter()
    try:
        await adapter.complete(ModelRequest(
            messages=(Message(role=MessageRole.USER, content="请只回复 OK。"),),
            model=adapter.default_model,
            max_output_tokens=64,
        ))
        return (perf_counter() - started) * 1000
    finally:
        await adapter.close()


def print_agent_event(event: AgentEvent) -> None:
    """把 Runtime 事件转换为简洁的终端进度信息。"""

    event_time = event.event_time.astimezone().strftime("%H:%M:%S")
    prefix = f"[{event_time}]"
    if event.type == 'agent_started':
        print(f"{prefix} Agent 开始执行")
    elif event.type == 'model_started':
        print(f"{prefix} 第 {event.step} 步：正在请求模型")
    elif event.type == 'model_completed':
        tool_count = len(event.message.tool_calls) if event.message else 0
        if tool_count:
            print(f"{prefix} 模型请求调用 {tool_count} 个工具")
        else:
            print(f"{prefix} 模型已返回回复")
    elif event.type == 'run_budget_warning':
        print(
            f"{prefix} Run 用量进入预警区："
            f"{event.run_budget_chargeable_tokens or 0} tokens · "
            f"{event.run_budget_model_calls or 0} calls"
        )
    elif event.type == 'run_budget_finalizing':
        print(f"{prefix} Run 用量达到收口线，正在生成最终答复")
    elif event.type == 'run_budget_exceeded':
        print(f"{prefix} Run 用量达到硬上限，停止继续请求模型")
    elif event.type == 'tool_started' and event.tool_call:
        print(f"{prefix} 开始执行工具：{event.tool_call.name}")
    elif event.type == 'tool_completed' and event.tool_result:
        status = "成功" if event.tool_result.success else "失败"
        print(
            f"{prefix} 工具 {event.tool_result.tool_name} {status} "
            f"({event.tool_result.duration_ms:.1f}ms)"
        )
    elif event.type == 'tool_approval_required' and event.tool_call:
        print(f"{prefix} 工具等待人工审批：{event.tool_call.name}")
    elif event.type == 'tool_approval_completed' and event.tool_call:
        decision = (
            event.approval_decision.value
            if event.approval_decision is not None
            else "unknown"
        )
        rule_text = (
            f" · 规则：{event.rule_description}"
            if event.rule_description is not None
            else ""
        )
        print(
            f"{prefix} 工具权限检查完成：{event.tool_call.name} · {decision}{rule_text}"
        )
    elif event.type == 'memory_reflection_started':
        print(f"{prefix} 正在整理本轮长期记忆")
    elif event.type == 'memory_reflection_completed':
        action = event.reflection_action or "none"
        suffix = (
            f" · {event.reflection_memory_id}"
            if event.reflection_memory_id is not None
            else ""
        )
        print(f"{prefix} 长期记忆整理完成：{action}{suffix}")
    elif event.type == 'memory_reflection_failed':
        print(f"{prefix} 长期记忆整理失败，已跳过")
    elif event.type == 'memory_reflection_skipped':
        reason = event.reflection_skip_reason or "policy"
        print(f"{prefix} 长期记忆整理已跳过：{reason}")
    elif event.type == 'memory_archive_started':
        print(f"{prefix} 长期记忆容量不足，正在选择可归档候选")
    elif event.type == 'memory_archive_completed':
        action = event.archive_action or "unknown"
        suffix = (
            f" · {event.archive_memory_id}"
            if event.archive_memory_id is not None
            else ""
        )
        print(f"{prefix} 长期记忆容量维护完成：{action}{suffix}")
    elif event.type == 'memory_archive_failed':
        print(f"{prefix} 长期记忆容量维护失败，未执行归档")
    elif event.type == 'agent_completed':
        print(f"{prefix} Agent 执行完成")
    elif event.type == 'agent_failed':
        reason = event.stop_reason.value if event.stop_reason else "unknown"
        print(f"{prefix} Agent 执行停止：{reason}")
    elif event.type == "memory_archive_skipped":
        print(f"{prefix} 长期记忆归档已跳过：{event.archive_skip_reason or 'policy'}")
    elif event.type == "skill_activated":
        print(f"{prefix} 技能已激活：{event.skill_name}")
    elif event.type == "skill_activation_failed":
        print(f"{prefix} 技能激活失败：{event.skill_name} · {event.skill_error or '未知原因'}")
    elif event.type == "agent_cancelled":
        print(f"{prefix} Agent 已取消")

def print_checkpoints(checkpoints: tuple[RunCheckpoint, ...]) -> None:
    """显示当前会话最近的恢复边界。"""

    if not checkpoints:
        print("当前会话暂无 Checkpoint。")
        return
    for checkpoint in checkpoints:
        updated_at = checkpoint.updated_at.astimezone().strftime("%m-%d %H:%M:%S")
        pending = len(checkpoint.pending_tool_calls)
        print(
            f"{checkpoint.run_id[:8]}  {updated_at}  "
            f"{checkpoint.status.value:<11} phase={checkpoint.phase.value} "
            f"step={checkpoint.step} pending_tools={pending}"
        )

def print_recovered_runs(runs: Sequence[Run]) -> None:
    """展示启动时修正的 Run 状态；不会在此恢复执行。"""
    for run in runs:
        if run.status == "interrupted":
            print(f"检测到中断 Run：{run.id[:8]} · 已保留恢复检查点")
        else:
            print(
                f"Run 状态修正：{run.id[:8]} → {run.status.value}"
                + (f" · {run.error}" if run.error else "")
            )

def print_trace(events: tuple[AgentEvent, ...]) -> None:
    """显示一次 Run 的完整事件时间线。"""

    for event in events:
        event_time = event.event_time.astimezone().strftime("%H:%M:%S.%f")[:-3]
        details: list[str] = []
        if event.step is not None:
            details.append(f"step={event.step}")
        if event.tool_call is not None:
            details.append(f"tool={event.tool_call.name}")
        if event.approval_decision is not None:
            details.append(f"decision={event.approval_decision.value}")
        if event.rule_id is not None:
            details.append(f"rule={event.rule_id[:8]}")
        if event.tool_result is not None:
            details.append(
                f"success={'true' if event.tool_result.success else 'false'}"
            )
        if event.type == 'model_started':
            if event.prepared_input_tokens is not None:
                details.append(f"input≈{event.prepared_input_tokens}")
            if event.tool_schema_tokens is not None:
                details.append(f"schemas≈{event.tool_schema_tokens}")
            if (
                event.tool_result_tokens_before is not None
                and event.tool_result_tokens_after is not None
            ):
                details.append(
                    "tool_results≈"
                    f"{event.tool_result_tokens_before}→"
                    f"{event.tool_result_tokens_after}"
                )
            if event.compaction_stage not in (None, "none"):
                details.append(f"context={event.compaction_stage}")
            if event.summary_provider is not None:
                details.append(
                    "summary="
                    f"{event.summary_provider}/{event.summary_model or 'default'}"
                )
            if event.summary_duration_ms is not None:
                details.append(f"summary_ms={event.summary_duration_ms:.1f}")
        if event.run_budget_status is not None:
            details.append(
                f"budget={event.run_budget_status}:"
                f"{event.run_budget_chargeable_tokens or 0}t/"
                f"{event.run_budget_model_calls or 0}calls"
            )
        detail_text = f"  {' '.join(details)}" if details else ""
        print(f"{event.sequence:03d}  {event_time}  {event.type.value}{detail_text}")

def print_permission_rules(rules: tuple[PermissionRule, ...]) -> None:
    """显示当前会话记住的工具审批规则。"""

    if not rules:
        print("当前会话没有已记住的审批规则。")
        return
    for rule in rules:
        created_at = rule.created_at.astimezone().strftime("%m-%d %H:%M:%S")
        print(f"{rule.id[:8]}  {created_at}  {rule.tool_name}  {rule.description}")

def print_memories(memories: Sequence[RegularMemoryRecord]) -> None:
    """输出适合终端快速浏览的长期记忆列表（Recall Cue）。"""

    if not memories:
        print("没有长期记忆。")
        return
    for memory in memories:
        cue = " ".join(memory.summary.split())
        print(f"{memory.id}  [{memory.status.value}]  {memory.title}")
        print(f"  Cue: {cue}")

def print_memory(memory: RegularMemoryRecord) -> None:
    """输出单条长期记忆的完整内容。"""

    print(
        f"ID: {memory.id}\n"
        f"状态: {memory.status.value}\n"
        f"标题: {memory.title}\n"
        f"访问次数: {memory.access_count}\n"
        f"创建时间: {memory.created_at.astimezone().isoformat()}\n"
        f"内容:\n{memory.render_full()}"
    )

def print_help() -> None:
    """仅列出 chat.py 当前已实现的交互命令。"""
    print(
        "可用命令：\n"
        "  /new [标题]       新建会话\n"
        "  /memories         查看记忆列表\n"
        "  /memory <记忆ID>  查看完整记忆\n"
        "  /permissions      查看当前会话的审批规则\n"
        "  /checkpoints      查看当前会话的检查点\n"
        "  /trace <Run ID>   查看运行事件\n"
        "  /clear            清空当前会话消息和摘要\n"
        "  /help             显示帮助\n"
        "  /exit 或 /quit    退出聊天"
    )


__all__ = ['print_banner', 'print_startup_status', 'print_conversation_divider', 'run_setup', 'print_agent_event', 'print_recovered_runs', 'print_memories', 'print_memory', 'print_permission_rules', 'print_checkpoints', 'print_trace', 'print_help']
