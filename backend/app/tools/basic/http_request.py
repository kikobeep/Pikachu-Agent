"""通用 HTTP 请求工具（需人工审核）。

网络访问存在 SSRF 风险，权限档位为 HUMAN_APPROVAL。
默认阻止解析到内网/回环地址的主机；可通过 ``allow_private`` 或
``allowed_hosts`` 显式放开（仅限本地/测试场景）。
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.models.types import ToolDefinition, ToolPermission

from ..base import BaseTool

MAX_HTTP_TIMEOUT_SECONDS = 60.0
_ALLOWED_METHODS = {"GET", "POST", "HEAD"}


class HttpRequestTool(BaseTool):
    def __init__(
        self,
        *,
        allow_private: bool = False,
        allowed_hosts: tuple[str, ...] = (),
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 200_000,
    ) -> None:
        self._allow_private = allow_private
        self._allowed_hosts = set(allowed_hosts)
        self._client = client
        self._max_response_bytes = max_response_bytes

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="http_request",
            description="向公网 URL 发送 HTTP GET、POST 或 HEAD 请求，返回状态码、响应头和响应正文。需要人工审批。",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要请求的 HTTP 或 HTTPS URL。",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "HEAD"],
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "description": "可选的请求头，各字段的值须为字符串。",
                    },
                    "body": {
                        "type": "string",
                        "description": "可选的请求正文，用于 POST 请求。",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": f"请求超时时间，单位为秒，上限为 {MAX_HTTP_TIMEOUT_SECONDS:g} 秒。",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        method = str(arguments.get("method", "GET")).upper()
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"'method' must be one of {sorted(_ALLOWED_METHODS)}")

        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("'url' must be a non-empty string")

        headers: dict[str, str] | None = None
        raw_headers = arguments.get("headers")
        if raw_headers is not None:
            if not isinstance(raw_headers, dict):
                raise ValueError("'headers' must be an object")
            headers = {str(key): str(value) for key, value in raw_headers.items()}

        body = arguments.get("body")
        if body is not None and not isinstance(body, str):
            raise ValueError("'body' must be a string")

        timeout = arguments.get("timeout_seconds", 15.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("'timeout_seconds' must be a positive number")
        timeout = min(float(timeout), MAX_HTTP_TIMEOUT_SECONDS)

        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("'url' must be an http(s) URL")

        if self._client is not None:
            return await self._do_request(
                self._client, method, url, headers, body, timeout
            )

        await self._assert_safe_url(parsed.hostname)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await self._do_request(
                client, method, url, headers, body, timeout
            )

    async def _do_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        content = bytearray()
        response_headers: dict[str, str] = {}
        status_code: int | None = None
        encoding = "utf-8"

        async with client.stream(
            method,
            url,
            headers=headers,
            content=body,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            response_headers = dict(response.headers.items())
            encoding = response.charset_encoding or "utf-8"
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) >= self._max_response_bytes:
                    break

        return {
            "method": method,
            "url": url,
            "status_code": status_code,
            "headers": response_headers,
            "text": bytes(content).decode(encoding, errors="replace"),
            "truncated": len(content) >= self._max_response_bytes,
            "elapsed_ms": round((perf_counter() - started_at) * 1000, 3),
        }

    async def _assert_safe_url(self, hostname: str) -> None:
        """SSRF 防护：阻止解析到内网/回环/保留地址的主机。"""
        if self._allow_private or hostname in self._allowed_hosts:
            return

        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
        except socket.gaierror:
            raise ValueError(
                f"URL host {hostname!r} could not be resolved (SSRF guard)."
            ) from None

        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if _is_blocked_address(ip):
                raise ValueError(
                    f"URL host {hostname!r} resolves to a private or internal "
                    "address and is blocked (SSRF guard)."
                )


def _is_blocked_address(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
