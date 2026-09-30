"""http_request 响应的类型感知压缩器。

http_request 的 output 是 JSON 字符串：``{method, url, status_code, headers, text,
truncated, elapsed_ms}``。压缩策略：① 裁剪 headers 只留关键响应头；② 对 ``text``
正文按类型路由——JSON 复用 compact_json，其余做头尾截断。无法解析时返回 None。
"""

from __future__ import annotations

import json

from .json import compact_json

_MAX_BODY_CHARS = 6000
_BODY_HEAD_CHARS = 4000
_BODY_TAIL_CHARS = 2000

# 只保留对 agent 有意义的响应头，丢弃 server/date/set-cookie/x-* 等噪声
_IMPORTANT_HEADERS = {
    "content-type",
    "content-length",
    "content-encoding",
    "content-language",
    "content-disposition",
    "location",
    "www-authenticate",
    "retry-after",
    "etag",
    "last-modified",
    "cache-control",
    "link",
}


def compact_http_response(text: str) -> str | None:
    """压缩 http_request 的 output（JSON 字符串）；无变化时返回 None。"""

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    changed = False
    new_data = dict(data)

    headers = data.get("headers")
    if isinstance(headers, dict) and headers:
        kept = {key: value for key, value in headers.items() if key.lower() in _IMPORTANT_HEADERS}
        if len(kept) < len(headers):
            new_data["headers"] = kept
            new_data["headers_omitted"] = len(headers) - len(kept)
            changed = True

    body = data.get("text")
    if isinstance(body, str) and body:
        compacted = _compact_body(body)
        if compacted is not None and len(compacted) < len(body):
            new_data["text"] = compacted
            changed = True

    if not changed:
        return None
    return json.dumps(new_data, ensure_ascii=False, separators=(",", ":"))


def _compact_body(body: str) -> str | None:
    # JSON 正文优先走结构压缩
    compacted_json = compact_json(body)
    if compacted_json is not None and len(compacted_json) < len(body):
        return compacted_json

    # 非 JSON（HTML / 纯文本）：头尾截断
    if len(body) <= _MAX_BODY_CHARS:
        return None
    head = body[: _BODY_HEAD_CHARS]
    tail = body[-_BODY_TAIL_CHARS:] if _BODY_TAIL_CHARS else ""
    omitted = len(body) - len(head) - len(tail)
    return f"{head}\n[... body truncated: omitted {omitted} characters]\n{tail}"


__all__ = ["compact_http_response"]
