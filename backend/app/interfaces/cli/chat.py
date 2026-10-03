
from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

from app.agent import Message, MessageRole
from app.agent.spec import load_agent_prompt
from app.conversation import Conversation,TriggerContext,ConversationSource
from app.conversation.service import ConversationService
from app.conversation.store import DEFAULT_DATABASE_PATH, ConversationStore
from app.context.tokens import default_token_estimator
from app.model import ModelProvider
from app.model.config import AgentMode
from app.tools.builtin.web_search import WebSearchTool
from app.tools.approval import ConsoleApprovalGate
from app.application import Application

from .cli_ui import (
    print_agent_event as _print_agent_event,
    print_assistant_message as _print_assistant_message,
    format_power_bar,
    print_banner,
    print_conversation_divider,
    print_help,
    print_startup_status,
    run_setup,
    save_search_api_key,
    print_checkpoints as _print_checkpoints,
    print_memories as _print_memories,
    print_memory as _print_memory,
    print_permission_rules as _print_permission_rules,
    print_recovered_runs as _print_recovered_runs,
    print_trace as _print_trace,
)

from app.task import (
    DEFAULT_TASKS_DIR,
)


class CliEventHandler:
    async def handle(self, event):
        from app.agent.events import AgentEventType

        if event.type is AgentEventType.TOOL_CONFIGURATION_REQUIRED:
            print("\n｜请填写TAVILY_API_KEY完成网络搜索，或直接回车不作网络搜索")
            api_key = await asyncio.to_thread(
                getpass.getpass,
                "TAVILY_API_KEY（不回显）：",
            )
            if api_key.strip():
                await asyncio.to_thread(save_search_api_key, api_key.strip())
                # 当前进程中的 SearchSettings 也会由 ToolRoundExecutor 更新；
                # 环境变量同步是为了后续新建的 settings 实例立即可见。
                import os

                os.environ["TAVILY_API_KEY"] = api_key.strip()
                print("已保存 TAVILY_API_KEY，正在重试网页搜索。")
                return api_key.strip()
            return None
        _print_agent_event(event)


def _initial_message(system_prompt: str | None) -> list[Message]:
    if not system_prompt:
        return []
    return [Message(role=MessageRole.SYSTEM, content=system_prompt)]


async def _run(args,*,offer_setup: bool = True,):
    # CLI 默认把执行 pikachu 时所在的目录作为当前 workspace。
    # Prompt 加载和 Application 必须使用同一个 workspace，避免 agent.md
    # 与文件工具、沙箱看到的工作区不一致。
    workspace_root = Path.cwd().resolve()
    effective_system_prompt = args.system or load_agent_prompt(
        explicit_path=args.agent_md,
        workspace_root=workspace_root,
    )
    try:
        app = Application(
            provider=args.provider,
            model=args.model,
            system_prompt=effective_system_prompt,
            database=args.database,
            tasks_dir=args.tasks_dir,
            # mcp_config=args.mcp_config,
            max_steps=args.max_steps,
            max_tool_rounds=args.max_tool_rounds,
            max_output_tokens=args.max_output_tokens,
            ace_enabled=args.ace,
            approval_gate=ConsoleApprovalGate(),
            workspace_root=workspace_root,
        )
    except ValueError as exc:
        # 不同配置入口可能抛出不同文案：既要识别“没有配置 Provider”，
        # 也要识别具体的“Provider 'openai' is not configured”。
        missing_provider = "not configured" in str(exc).lower()
        if (
            offer_setup     # 启动时如果发现没有配置模型，是否允许自动进入设置向导
            and missing_provider
            and args.message is None
            and sys.stdin.isatty()
        ):
            print("尚未配置主模型，正在进入首次设置。")
            if await run_setup():
                return await _run(args, offer_setup=False)
            return 0
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "请重新运行 `pikachu` 进入配置向导，或运行 `pikachu --setup` 手动配置。",
            file=sys.stderr,
        )
        return 2
    provider = app.provider
    model = app.model
    await app.start()
    try:
        conversation_store = app.conversation_store
        conversation_service = app.conversation_service
        # automation_scheduler = app.automation_scheduler
        # mcp_manager = app.mcp_manager
        tool_registry = app.tool_registry

        try:
            conversation, history, resumed = await _load_or_create_conversation(
                conversation_store,
                identifier=args.conversation,
                force_new=args.new_conversation,
                system_prompt=effective_system_prompt,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    
        action = "会话已恢复" if resumed else "新会话已创建"
        try:
            search_tool = tool_registry.get("web_search")
        except KeyError:
            # 保持对旧注册表实现的兼容；当前实现始终注册 web_search。
            search_tool = None
        search_status = "未启用"
        if isinstance(search_tool, WebSearchTool):
            if not search_tool.is_configured:
                search_status = "Tavily · 待配置 API Key"
            elif search_tool.provider_name == "tavily":
                search_status = "Tavily"
            else:
                search_status = "DuckDuckGo · 配置 TAVILY_API_KEY 可启用 Tavily"

        reflection_status = "启用" if app.memory_reflection_enabled else "关闭"
        reflection_model = app.memory_reflector.model_hint or "未解析"
        reflection_provider = app.memory_reflector.provider_hint or "未解析"

        context_window = app.context_manager.context_window_for(provider, model)
        visible_tools = tuple(
            tool_registry.definitions_for_mode(AgentMode.DEFAULT)
        )
        context_used = default_token_estimator().estimate_request(
            history,
            tools=visible_tools,
            model=model,
            provider=provider,
        )
        print_banner(
            status=(
                ("Model", f"{provider}/{model}"),
                ("Power", format_power_bar(context_used, context_window)),
                ("Context", f"{context_used:,} / {context_window:,}"),
                ("Mode", "exploring"),
            )
        )

        print_startup_status(
            (
                ("主模型", f"{provider}/{model}"),
                (
                    "会话",
                    f"{action} {conversation.id[:8]} · {conversation.title} · "
                    f"{conversation.message_count} 条消息",
                ),
                ("搜索", search_status),
                # ("MCP", mcp_status),
                (
                    "长期记忆",
                    f"Sparse Memory · 反思{reflection_status} "
                    f"{reflection_provider}/{reflection_model}",
                ),
                # ("技能学习", learning_status),
            )
            # notices=notices,
        )
        print_conversation_divider(
            f"{conversation.title} · {conversation.id[:8]}"
        )

        if app.reconciled_runs:
            _print_recovered_runs(app.reconciled_runs)
    
        # 模式一：发送一条消息后退出。
        if args.message is not None:
            success, conversation = await _send_message(
                conversation_service=conversation_service,
                conversation_store=conversation_store,
                conversation=conversation,
                provider=provider,
                history=history,
                content=args.message,
                model=model,
            )
            await _skill_mining(skill_miner)
            return 0 if success else 1

        # 模式二：持续读取用户输入。
        return await _run_interactive(
            app=app,
            conversation=conversation,
            history=history,
            system_prompt=effective_system_prompt,
        )
    finally:
        await app.close()


async def _run_interactive(*, app, conversation, history, system_prompt) -> int:
    """读取输入：命令交给命令处理函数，普通消息交给会话服务。"""
    while True:
        try:
            content = (await asyncio.to_thread(input, "\n你> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n聊天已结束。")
            return 0

        if not content:
            continue
        if content in {"/exit", "/quit"}:
            print("聊天已结束。")
            return 0

        handled, conversation, history = await _handle_command(
            content,
            app=app,
            conversation=conversation,
            history=history,
            system_prompt=system_prompt,
        )
        if handled:
            continue

        _, conversation = await _send_message(
            conversation_service=app.conversation_service,
            conversation_store=app.conversation_store,
            conversation=conversation,
            provider=app.provider,
            history=history,
            content=content,
            model=app.model,
        )


async def _handle_command(content, *, app, conversation, history, system_prompt):
    """处理已有 CLI 命令，返回是否已处理以及最新的会话、历史。"""
    conversation_store = app.conversation_store
    summary_store = app.summary_store
    memory_manager = app.memory_manager
    rule_store = app.rule_store
    checkpoint_store = app.checkpoint_store
    trace_store = app.trace_store

    if content == "/new" or content.startswith("/new "):
        title = content.removeprefix("/new").strip() or "新会话"
        conversation = await conversation_store.create(
            title=title,
            messages=_initial_message(system_prompt),
        )
        history = list(await conversation_store.load_messages(conversation.id))
        print(f"已创建会话：{conversation.id[:8]} · {conversation.title}")
        print_conversation_divider(
            f"{conversation.title} · {conversation.id[:8]}"
        )
        return True, conversation, history

    if content == "/memories" or content.startswith("/memories "):
        _print_memories(await memory_manager.list())
        return True, conversation, history

    if content == "/memory" or content.startswith("/memory "):
        identifier = content.removeprefix("/memory").strip()
        if not identifier:
            print("用法：/memory <记忆ID>")
            return True, conversation, history
        memory = await memory_manager.read(identifier)
        if memory is None:
            print(f"找不到记忆：{identifier}")
            return True, conversation, history
        _print_memory(memory)
        return True, conversation, history
    if content == "/permissions":
        rules = await rule_store.list(scope_ids=(conversation.id,))
        _print_permission_rules(rules)
        return True, conversation, history
    if content == "/checkpoints":
        _print_checkpoints(
            await checkpoint_store.list(
                conversation_id=conversation.id,
            )
        )
        return True, conversation, history
    if content == "/clear":
        history = _initial_message(system_prompt)
        conversation = await conversation_store.replace_messages(
            conversation.id,
            history,
        )
        await summary_store.delete(conversation.id)
        print("上下文已清空。")
        return True, conversation, history
    if content == "/trace" or content.startswith("/trace "):
        identifier = content.removeprefix("/trace").strip()
        if not identifier:
            print("用法：/trace <Run ID>")
            return True, conversation, history
        try:
            run = await trace_store.resolve(identifier)
        except ValueError as exc:
            print(exc)
            return True, conversation, history
        if run is None:
            print(f"找不到 Run：{identifier}")
            return True, conversation, history
        print(
            f"Run {run.run_id} · {run.status.value} · {run.event_count} 个事件"
        )
        _print_trace(await trace_store.load_events(run.run_id))
        return True, conversation, history
    if content == "/help":
        print_help()
        return True, conversation, history

    return False, conversation, history


async def _send_message(
    *,
    conversation_service: ConversationService,
    conversation_store: ConversationStore,
    conversation: Conversation,
    provider: ModelProvider | str,
    history: list[Message],
    content: str,
    model: str,
) -> tuple[bool, Conversation]:

    try:
        dispatch = await conversation_service.dispatch(
            conversation_id=conversation.id,
            content=content,
            trigger=TriggerContext(source=ConversationSource.MANUAL),
            event_handler=CliEventHandler(),
        )
    except KeyboardInterrupt:
        print("\n[cancel] 已取消当前 Run。")
        return False, conversation

    result = dispatch.result
    
    history[:] = result.messages
    if conversation.title == "新会话":
        conversation = await conversation_store.rename(
            conversation.id,
            " ".join(content.split()).strip()[:50] # new title
        )
    stop_reason = dispatch.run.stop_reason or result.stop_reason.value
    provider_name = provider.value if isinstance(provider, ModelProvider) else provider
    print(
        f"\n[{provider_name}/{model} · {result.steps} steps · "
        f"{result.usage.total_tokens} tokens · {stop_reason}]"
    )
    if result.tool_calls:
        tools = ", ".join(
            f"{record.tool_call.name}({'成功' if record.result.success else '失败'})"
            for record in result.tool_calls
        )
        print(f"[工具调用：{tools}]")
    if result.final_message.content:
        _print_assistant_message(result.final_message.content)
    if dispatch.run.status.value == "cancelled":
        print("[Run 已被取消]")
    return result.ok, conversation


async def _load_or_create_conversation(
    store: ConversationStore,
    *,
    identifier: str | None,
    force_new: bool,
    system_prompt: str | None,
) -> tuple[Conversation, list[Message], bool]:
    """加载指定或最近会话；不存在时创建新会话。"""

    if force_new:
        conversation = await store.create(messages=_initial_message(system_prompt))
        return conversation, list(await store.load_messages(conversation.id)), False

    if identifier:
        conversation = await store.resolve(identifier)
        if conversation is None:
            raise ValueError(f"找不到会话：{identifier}")
    else:
        conversation = await store.latest()

    if conversation is None:
        conversation = await store.create(messages=_initial_message(system_prompt))
        return conversation, list(await store.load_messages(conversation.id)), False

    history = list(await store.load_messages(conversation.id))
    return conversation, history, True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pikachu",
        description="启动 CLI，或完成首次模型设置。",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=[provider.value for provider in ModelProvider],
        help="指定本次使用的模型 Provider；默认使用设置中的主模型。",
    )
    parser.add_argument("-m", "--model", help="临时覆盖模型名称。")
    parser.add_argument(
        "--message",
        help="发送一条消息后退出，不进入交互聊天。",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="启动交互式模型设置，密钥优先保存到 macOS Keychain。",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="覆盖本次会话的系统提示词；未指定时从 agent.md 加载。",
    )
    parser.add_argument(
        "--agent-md",
        type=Path,
        default=None,
        help="指定 agent.md 路径；未指定时自动查找 <workspace>/agent.md → ~/.sidekick/agent.md → 内置默认。",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="每次模型回复允许生成的最大 Token；默认使用 Provider 配置。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="每条消息最多执行的模型/工具循环步数。",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=15,
        help="强制模型收口最终回答前允许的最大工具轮数。",
    )
    parser.add_argument(
        "--ace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 ACE playbook；默认关闭，可用 --no-ace 显式关闭。",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="会话 SQLite 数据库路径。",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="持久化 Task JSON 文件目录。",
    )
    # parser.add_argument(
    #     "--mcp-config",
    #     type=Path,
    #     default=DEFAULT_MCP_CONFIG_PATH,
    #     help="MCP Server JSON 配置文件路径。",
    # )
    parser.add_argument(
        "--conversation",
        help="使用完整会话 ID 或唯一前缀恢复会话。",
    )
    parser.add_argument(
        "--new",
        "--new-conversation",
        dest="new_conversation",
        action="store_true",
        help="新建会话，不恢复最近会话。",
    )
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds must be greater than zero")
    if args.conversation and args.new_conversation:
        parser.error("--conversation and --new-conversation cannot be used together")
    if args.setup and (args.message or args.conversation or args.new_conversation):
        parser.error("--setup 不能和会话或单次消息参数同时使用")
    return args

async def _main(args: argparse.Namespace) -> int:
    if args.setup:
        should_start = await run_setup()
        if not should_start:
            return 0
    return await _run(args)


def main() -> None:
    raise SystemExit(asyncio.run(_main(_parse_args())))


if __name__ == "__main__":
    main()
