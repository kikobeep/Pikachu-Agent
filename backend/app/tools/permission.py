"""根据工具声明的权限生成执行决定。"""

from dataclasses import dataclass

from .approval import ApprovalRequest
from .base import ToolExecutionContext, ToolPermission


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
    denied_reason: str | None = None
    approval_request: ApprovalRequest | None = None


class PermissionHook:
    async def before_execute(self, context: ToolExecutionContext) -> ToolHookDecision | None:
        definition = context.tool_definition
        if definition is None:
            return ToolHookDecision(denied_reason="Missing tool definition")
        if definition.permission is ToolPermission.FORBIDDEN:
            return ToolHookDecision(denied_reason=f"Tool {definition.name!r} is forbidden")
        if definition.permission is ToolPermission.ALLOWED:
            return None
        if definition.permission is not ToolPermission.HUMAN_APPROVAL:
            return ToolHookDecision(denied_reason="Unknown tool permission")
        if context.arguments is None:
            return ToolHookDecision(denied_reason="Missing parsed tool arguments")
        return ToolHookDecision(approval_request=ApprovalRequest(
            tool_call_id=context.tool_call.id,
            tool_name=context.tool_call.name,
            arguments=context.arguments,
            description=definition.description,
            run_id=context.run_id,
            conversation_id=context.conversation_id,
        ))
