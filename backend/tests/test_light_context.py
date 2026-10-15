"""LIGHT MODE accepts native tool calls even when the model sends no text."""

from app.agent.light.context import parse_model_output
from app.model.config import ToolCall


def test_native_tool_call_without_text_is_normalized() -> None:
    call = ToolCall(id="call-1", name="get_current_time", arguments={})

    envelope = parse_model_output(None, native_tool_calls=(call,))

    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].id == "call-1"
    assert envelope.tool_calls[0].name == "get_current_time"
    assert envelope.tool_calls[0].arguments == {}
