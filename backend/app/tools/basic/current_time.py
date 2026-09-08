"""按需查询当前时间的只读工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..base import BaseTool, ToolDefinition, ToolPermission


class CurrentTimeTool(BaseTool):
    """返回系统本地时间或指定 IANA 时区的当前时间。"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_current_time",
            description=(
                "按需获取当前实际日期和时间。回答涉及今天、明天、昨天、"
                "现在、近期日期、截止时间或相对时间的问题前，应先调用此工具。"
                "未指定时区时使用进程所在系统的本地时区。"
                "此工具只读，无需人工审批。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "可选的 IANA 时区，例如 Asia/Shanghai 或 America/New_York。"
                        ),
                    }
                },
                "additionalProperties": False,
            },
            permission=ToolPermission.ALLOWED,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timezone = arguments.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            raise TypeError("timezone 必须是字符串")

        if timezone is None or not timezone.strip():
            current = datetime.now().astimezone()
            timezone_name = _local_timezone_name(current)
        else:
            normalized = timezone.strip()
            try:
                zone = ZoneInfo(normalized)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"未知 IANA 时区: {normalized}") from exc
            current = datetime.now(zone)
            timezone_name = normalized

        return {
            "datetime": current.isoformat(timespec="seconds"),
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "timezone": timezone_name,
            "utc_offset": current.strftime("%z")[:3] + ":" + current.strftime("%z")[3:],
            "unix_timestamp": int(current.timestamp()),
        }


def _local_timezone_name(current: datetime) -> str:
    key = getattr(current.tzinfo, "key", None)
    if isinstance(key, str) and key:
        return key
    return current.tzname() or str(current.tzinfo)


__all__ = ["CurrentTimeTool"]
