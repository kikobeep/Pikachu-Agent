
from __future__ import annotations

from collections import Counter

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.model.config import Message, MessageRole


class BlockType(StrEnum):
    SYSTEM = "system"
    CONVERSATION = "conversation"
    TOOL_ROUND = "tool_round"
    ORPHAN_TOOL = "orphan_tool"


@dataclass(frozen=True)
class MessageBlock:
    """块基类：一序列消息的不可变分组。"""

    messages: tuple[Message, ...]

    @property
    def block_type(self) -> BlockType:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.messages)


@dataclass(frozen=True)
class SystemBlock(MessageBlock):
    """系统提示块。"""

    @property
    def block_type(self) -> BlockType:
        return BlockType.SYSTEM


@dataclass(frozen=True)
class ConversationBlock(MessageBlock):
    """一轮普通对话块（user + 无工具的 assistant）。"""

    @property
    def block_type(self) -> BlockType:
        return BlockType.CONVERSATION

@dataclass(frozen=True)
class OrphanToolBlock(MessageBlock):
    """未完成或协议异常的工具消息块；压缩时必须保守保留。"""

    reason: str = "malformed tool protocol"

    @property
    def block_type(self) -> BlockType:
        return BlockType.ORPHAN_TOOL

def partition_messages(
    messages: Sequence[Message],
) -> tuple[MessageBlock, ...]:

    blocks: list[MessageBlock] = []
    conversation: list[Message] = []

    def flush_conversation() -> None:
        nonlocal conversation
        if conversation:
            blocks.append(ConversationBlock(tuple(conversation)))
            conversation = []

    index = 0
    count = len(messages)
    while index < count:
        message = messages[index]
        if message.role is MessageRole.SYSTEM:
            flush_conversation()
            system = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.SYSTEM:
                system.append(messages[index])
                index += 1
            blocks.append(SystemBlock(tuple(system)))
            continue
        if message.role is MessageRole.USER:
            flush_conversation()
            conversation.append(message)
            index += 1
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            flush_conversation()
            tool_round = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.TOOL:
                tool_round.append(messages[index])
                index += 1
            try:
                blocks.append(ToolRoundBlock(tuple(tool_round)))
            except ValueError as exc:
                blocks.append(
                    OrphanToolBlock(tuple(tool_round), reason=str(exc))
                )
            continue
        if message.role is MessageRole.TOOL:
            flush_conversation()
            orphan_results = [message]
            index += 1
            while index < count and messages[index].role is MessageRole.TOOL:
                orphan_results.append(messages[index])
                index += 1
            blocks.append(
                OrphanToolBlock(
                    tuple(orphan_results),
                    reason="orphan tool result without assistant tool call",
                )
            )
            continue
        # 无工具的 assistant 并入当前对话轮。
        conversation.append(message)
        index += 1

    flush_conversation()
    return tuple(blocks)


@dataclass(frozen=True)
class ToolRoundBlock(MessageBlock):
    """一轮完整且 ID 对应关系合法的工具调用块。"""

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ToolRoundBlock messages cannot be empty")
        assistant = self.messages[0]
        if assistant.role is not MessageRole.ASSISTANT or not assistant.tool_calls:
            raise ValueError(
                "ToolRoundBlock must start with an assistant tool call message"
            )

        expected_ids = [call.id for call in assistant.tool_calls]
        if any(not call_id for call_id in expected_ids):
            raise ValueError("ToolCall id cannot be empty")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("ToolCall ids must be unique within one tool round")

        result_messages = self.messages[1:]
        if any(message.role is not MessageRole.TOOL for message in result_messages):
            raise ValueError("ToolRoundBlock may only contain trailing tool results")
        result_ids = [message.tool_call_id for message in result_messages]
        if any(not tool_call_id for tool_call_id in result_ids):
            raise ValueError("ToolResult message requires tool_call_id")
        if Counter(result_ids) != Counter(expected_ids):
            raise ValueError(
                "ToolResult tool_call_id values must exactly match ToolCall ids"
            )

    @property
    def block_type(self) -> BlockType:
        return BlockType.TOOL_ROUND

    '''
    SystemBlock：
  [系统提示]
  
    ConversationBlock：
  [用户：帮我搜索 asyncio]

    ToolRoundBlock：
  [助手的工具调用, 工具结果]

    ConversationBlock：
  [助手：这是总结]

    ConversationBlock：
  [用户：再举个例子, 助手：这里有一个例子]
    '''
