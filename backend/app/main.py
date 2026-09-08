import argparse



def parse_args():
    parse = argparse.ArgumentParser(prog="python -m app")
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
    parser.add_argument(
        "--mcp-config",
        type=Path,
        default=DEFAULT_MCP_CONFIG_PATH,
        help="MCP Server JSON 配置文件路径。",
    )
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
    return args





def main():
    args = parse_args()