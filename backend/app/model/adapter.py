"""模型适配器的抽象接口。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing import Any

from ..types import Message, MessageRole, ModelUsage, ToolCall
from ..tools.base import ToolDefinition
from .config import ProviderConfig
from .type import ApiStyle, ModelRequest, ModelResponse


@dataclass(slots=True)
class _StreamStatus:
    """单次流式请求的重试状态；每次重试必须创建新实例。"""

    stream_opened: bool = False
    visible_delta_emitted: bool = False
    terminal_event_received: bool = False

    @property
    def can_retry(self) -> bool:
        # 仅重试尚未产生可见输出、也未收到终止事件的中断流。
        return (
            self.stream_opened
            and not self.visible_delta_emitted
            and not self.terminal_event_received
        )


class ModelAdapter(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def default_model(self) -> str:
        return self.config.model

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """返回一次统一格式的模型响应。"""

    @abstractmethod
    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        """流式通知增量，并返回完整响应。"""

    @abstractmethod
    async def close(self) -> None:
        """释放模型提供商客户端资源。"""


class ModelCompatibleAdapter(ModelAdapter):
    '''
    gpt & deepseek
    '''
    def __init__(
        self,
        config: ProviderConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(config)
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=config.api_key_value(),
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        self._client = client


    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            if self.config.api_style is ApiStyle.RESPONSES:
                return await self._complete_response(request)
            return await self._complete_chat(request)
        except Exception as exc:
            raise RuntimeError(f"{self.provider} model request failed: {exc}") from exc

    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        """使用 Provider 原生流，同时在结束时还原完整 ``ModelResponse``。"""

        retries = 0
        while True:
            status = _StreamStatus()
            try:
                if self.config.api_style is ApiStyle.RESPONSES:
                    return await self._stream_responses(
                        request,
                        on_text_delta,
                        on_reasoning_delta,
                        status=status,
                    )
                return await self._stream_chat(
                    request, on_text_delta, on_reasoning_delta, status=status,
                )
            except Exception as exc:
                can_retry = (
                    status.can_retry and retries < self.config.max_retries
                )
                if can_retry:
                    retries += 1
                    continue

                raise RuntimeError(
                    f"{self.provider} model stream failed: {exc}"
                ) from exc

    async def close(self) -> None:
        await self._client.close()


    async def _complete_response(
        self,
        request: ModelRequest
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "input": _parse_response_message(request.messages)
        }
        if request.tools:
            kwargs["tools"] = [_parse_response_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        response = await self._client.responses.create(**kwargs)
        return _parse_response(response, self.provider)

    async def _complete_chat(
        self,
        request: ModelRequest
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [_parse_chat_message(message) for message in request.messages]
        }

        if request.tools:
            kwargs["tools"] = [_parse_chat_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body
        response = await self._client.chat.completions.create(**kwargs)
        return _parse_chat(response,self.provider)

    async def _stream_chat(
        self,
        request: ModelRequest,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        *,
        status: _StreamStatus,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [_parse_chat_message(message) for message in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            kwargs["tools"] = [_parse_chat_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        events = await self._client.chat.completions.create(**kwargs)
        status.stream_opened = True
        response_id = None
        model = request.model or self.default_model
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # 多个工具调用可能交错到达，只能按 index 分别累积。
        calls: dict[int, dict[str, str]] = {}
        finish_reason = None
        usage = None
        async for chunk in events:
            response_id = chunk.id or response_id
            model = chunk.model or model
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                # 统一响应只承载一个候选，与非流式路径保持一致。
                if choice.index != 0:
                    continue
                delta = choice.delta
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                    status.terminal_event_received = True
                for text in (delta.content, getattr(delta, "refusal", None)):
                    if text:
                        text_parts.append(text)
                        status.visible_delta_emitted = True
                        await on_text_delta(text)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if on_reasoning_delta is not None:
                        status.visible_delta_emitted = True
                        await on_reasoning_delta(reasoning)
                for call in delta.tool_calls or ():
                    accumulated = calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        accumulated["id"] += call.id
                    if call.function is not None:
                        accumulated["name"] += call.function.name or ""
                        accumulated["arguments"] += call.function.arguments or ""
            # finish_reason 后可能还有 choices=[] 的用量分片，不能提前 break。

        if finish_reason is None or response_id is None:
            raise RuntimeError("Chat stream ended without a final choice")
        tool_calls = []
        for index in sorted(calls):
            call = calls[index]
            if not call["id"] or not call["name"]:
                raise RuntimeError(f"Chat stream returned an incomplete tool call at index {index}")
            tool_calls.append(ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=_parse_arguments(call["arguments"]),
            ))
        return ModelResponse(
            id=response_id,
            provider=self.provider,
            model=model,
            response=Message(
                role=MessageRole.ASSISTANT,
                content="".join(text_parts) or None,
                tool_calls=tuple(tool_calls),
                reasoning="".join(reasoning_parts) or None,
            ),
            finish_reason=finish_reason,
            usage=_chat_usage(usage),
            raw={},
        )
       

    async def _stream_responses(
        self,
        request: ModelRequest,
        on_text_delta: Callable[[str], Awaitable[None]],
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        *,
        status: _StreamStatus,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "input": _parse_response_message(request.messages),
            "stream": True
        }
        if request.tools:
            kwargs["tools"] = [_parse_response_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        events = await self._client.responses.create(**kwargs)
        status.stream_opened = True
        final_response = None
    
        async for event in events:
            event_type = getattr(event, "type", None)
            if event_type in ("response.output_text.delta", "response.refusal.delta"):
                delta = getattr(event, "delta", "")
                if delta:
                    # 回调可能已输出内容后才抛错，必须在调用前禁止重试。
                    status.visible_delta_emitted = True
                    await on_text_delta(delta)
            elif event_type == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "")
                if delta and on_reasoning_delta is not None:
                    status.visible_delta_emitted = True
                    await on_reasoning_delta(delta)
            elif event_type in ("response.completed", "response.incomplete"):
                status.terminal_event_received = True
                final_response = getattr(event, "response", None)
                break
            elif event_type in ("response.failed", "error"):
                status.terminal_event_received = True
                error = getattr(getattr(event, "response", None), "error", None)
                message = getattr(error, "message", None) or getattr(event, "message", None)
                raise RuntimeError(message or f"Stream ended with {event_type}")

        if final_response is None:
            raise RuntimeError("Responses stream ended without a final response")
        return _parse_response(final_response, self.provider)
    


def _parse_response_message(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.TOOL:
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content or "",
                }
            )
            continue
        if message.content:
            result.append(
                {
                    "role": message.role.value,
                    "content": message.content
                }
            )
        for call in message.tool_calls:
            result.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": (call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)),
                }
            )
    return result



def _parse_chat_message(message: Message) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": (call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)),
                },
            }
            for call in message.tool_calls
        ]
    return result


def _parse_chat_tool(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if getattr(tool, "strict", None) is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}

def _parse_response_tool(tool: ToolDefinition) -> dict[str, Any]:
    parse_tool: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if getattr(tool, "strict", None) is not None:
        parse_tool["strict"] = tool.strict
    return parse_tool

def _parse_response(response: Any, provider: str) -> ModelResponse:
    tool_calls = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            tool_calls.append(ToolCall(
                id=item.call_id,
                name=item.name,
                arguments=_parse_arguments(item.arguments),
            ))

    return ModelResponse(
        id=response.id,
        provider=provider,
        model=response.model,
        response=Message(
            role=MessageRole.ASSISTANT,
            content=response.output_text or None,
            tool_calls=tool_calls,
            reasoning=_parse_reasoning(response),
        ),
        finish_reason=_parse_finish_reason(response, tool_calls),
        usage=_parse_usage(getattr(response, "usage", None)),
        raw=response.model_dump(mode="json"),
    )

def _parse_chat(response: Any, provider: str) -> ModelResponse:
    choice = response.choices[0]
    message = choice.message

    tool_calls = tuple(
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments),
        )
        for call in (message.tool_calls or ())
    )

    return ModelResponse(
        id=response.id,
        provider=provider,
        model=response.model,
        response=Message(
            role=MessageRole.ASSISTANT,
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            reasoning=getattr(message, "reasoning_content", None),
        ),
        finish_reason=choice.finish_reason,
        usage=_chat_usage(getattr(response, "usage", None)),
        raw=response.model_dump(mode="json"),
    )

def _parse_reasoning(response: Any) -> str | None:
    """Responses API 的推理摘要（reasoning items 的 summary 拼接）。"""
    parts: list[str] = []
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) == "reasoning":
            for summary in getattr(item, "summary", ()):
                parts.append(summary.text)
    return "".join(parts) or None

def _parse_finish_reason(response: Any, tool_calls: list[ToolCall]) -> str:
    if tool_calls:
        return 'tool_calls'
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return getattr(details, "reason", None) or "incomplete"
    return getattr(response, "status", None) or "stop"

def _parse_usage(usage: Any) -> ModelUsage:
    return _usage(usage, "input_tokens", "output_tokens", "input_tokens_details")


def _chat_usage(usage: Any) -> ModelUsage:
    return _usage(usage, "prompt_tokens", "completion_tokens", "prompt_tokens_details")


def _usage(usage: Any, input_field: str, output_field: str, details_field: str) -> ModelUsage:
    input_tokens = int(getattr(usage, input_field, 0) or 0)
    output_tokens = int(getattr(usage, output_field, 0) or 0)
    details = getattr(usage, details_field, None)
    cached_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached_tokens is None:
        cached_tokens = getattr(details, "cached_tokens", None)
    if cached_tokens is None:
        cached_tokens = getattr(usage, "cached_tokens", None)
    if cached_tokens is not None:
        cached_tokens = max(0, int(cached_tokens))
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(getattr(usage, "total_tokens", None) or input_tokens + output_tokens),
        cached_input_tokens=cached_tokens,
        uncached_input_tokens=(
            max(0, input_tokens - cached_tokens) if cached_tokens is not None else None
        ),
        cache_read_input_tokens=cached_tokens,
        cache_write_input_tokens=getattr(details, "cache_creation_input_tokens", None),
        model_calls=1,
    )


def _parse_arguments(arguments: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return parsed if isinstance(parsed, dict) else arguments




async def main() -> None:
    import argparse
    from .config import ModelConfig

    parser = argparse.ArgumentParser(description="手动测试模型调用")
    parser.add_argument("--provider", choices=("openai", "qwen", "deepseek"),
                        help="默认使用 MODEL_DEFAULT_PROVIDER")
    parser.add_argument("--stream", action="store_true", help="实时打印流式输出")
    parser.add_argument("--config-only", action="store_true", help="仅检查配置，不发送请求")
    args = parser.parse_args()

    settings = ModelConfig()
    config = settings.load_provider_config(args.provider or settings.model_default_provider)
    print(f"provider={config.provider}, model={config.model}, api_style={config.api_style.value}")
    if args.config_only:
        return

    request = ModelRequest(
        messages=(
            Message(
                role=MessageRole.USER,
                content="一件商品先涨价20%，再降价20%，最终售价是96元。"
                        "请问原价是多少？请用两行以内给出计算过程和答案。",
            ),
        ),
        max_output_tokens=8192,
    )

    async def on_text(text: str) -> None:
        print(text, end="", flush=True)

    adapter = ModelCompatibleAdapter(config)
    try:
        if args.stream:
            result = await adapter.complete_stream(request, on_text_delta=on_text)
            print()
        else:
            result = await adapter.complete(request)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        await adapter.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
