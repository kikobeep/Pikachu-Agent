"""json_array / json_object 输出的类型感知压缩器。

数组：保留前 1/3 + 后 1/4 + 高价值项 + 均匀采样；对象：保留重要 key（error/status/
message 等）+ 高价值 value；递归截断长字符串与嵌套结构。用 ``_compact`` 元数据包裹，
标明原始/保留/省略数量。无法解析成 list/dict 时返回 None。
"""

from __future__ import annotations

import json
from typing import Any

_MAX_ARRAY_ITEMS = 12
_MAX_OBJECT_KEYS = 24
_MAX_STRING_CHARS = 240
_MAX_NESTED_ITEMS = 6

_IMPORTANT_KEYS = {
    "error",
    "errors",
    "exception",
    "failed",
    "failure",
    "message",
    "reason",
    "status",
    "summary",
    "traceback",
    "warning",
    "warnings",
}


def compact_json(text: str) -> str | None:
    """压缩 JSON 输出；不是合法 JSON 或不是 list/dict 时返回 None。"""

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if isinstance(parsed, list):
        result = _compact_array(parsed)
    elif isinstance(parsed, dict):
        result = _compact_object(parsed)
    else:
        return None

    if result is None:
        return None
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_array(items: list[Any]) -> dict | None:
    if not items:
        return None

    selected = _select_array_indexes(items, max_items=_MAX_ARRAY_ITEMS)
    kept = [_summarize_value(items[index]) for index in selected]
    return {
        "_compact": {
            "type": "json_array",
            "original_items": len(items),
            "kept_items": len(kept),
            "omitted_items": len(items) - len(kept),
            "schema_keys": _schema_keys(items),
        },
        "items": kept,
    }


def _compact_object(value: dict[str, Any]) -> dict | None:
    if not value:
        return None

    selected_keys = _select_object_keys(value, max_keys=_MAX_OBJECT_KEYS)
    compacted = {key: _summarize_value(value[key]) for key in selected_keys}
    return {
        "_compact": {
            "type": "json_object",
            "original_keys": len(value),
            "kept_keys": len(compacted),
            "omitted_keys": len(value) - len(compacted),
            "omitted_key_names": [key for key in value.keys() if key not in compacted][:20],
        },
        "object": compacted,
    }


def _select_array_indexes(items: list[Any], *, max_items: int) -> list[int]:
    if len(items) <= max_items:
        return list(range(len(items)))

    selected: set[int] = set()
    front = max(1, max_items // 3)
    back = max(1, max_items // 4)
    for index in range(min(front, len(items))):
        selected.add(index)
    for index in range(max(0, len(items) - back), len(items)):
        selected.add(index)

    scored = sorted(
        ((index, _value_score(item)) for index, item in enumerate(items)),
        key=lambda pair: (pair[1], -pair[0]),
        reverse=True,
    )
    for index, score in scored:
        if len(selected) >= max_items:
            break
        if score > 0:
            selected.add(index)

    cursor = front
    while len(selected) < max_items and cursor < len(items):
        selected.add(cursor)
        cursor += max(1, len(items) // max_items)

    return sorted(selected)


def _select_object_keys(value: dict[str, Any], *, max_keys: int) -> list[str]:
    keys = list(value.keys())
    if len(keys) <= max_keys:
        return keys

    selected: list[str] = []
    for key in keys:
        if _is_important_key(key) or _value_score(value[key]) > 0:
            selected.append(key)
        if len(selected) >= max_keys:
            return selected

    for key in keys:
        if key not in selected:
            selected.append(key)
        if len(selected) >= max_keys:
            break
    return selected


def _summarize_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_CHARS:
            return value
        return value[: _MAX_STRING_CHARS].rstrip() + "...[truncated]"
    if isinstance(value, list):
        if len(value) <= _MAX_NESTED_ITEMS:
            return [_summarize_value(item) for item in value]
        selected = _select_array_indexes(value, max_items=_MAX_NESTED_ITEMS)
        return {
            "_type": "array",
            "items": len(value),
            "kept": [_summarize_value(value[index]) for index in selected],
            "omitted": len(value) - len(selected),
        }
    if isinstance(value, dict):
        if len(value) <= _MAX_NESTED_ITEMS:
            return {key: _summarize_value(item) for key, item in value.items()}
        selected_keys = _select_object_keys(value, max_keys=_MAX_NESTED_ITEMS)
        return {
            "_type": "object",
            "keys": len(value),
            "kept": {key: _summarize_value(value[key]) for key in selected_keys},
            "omitted": len(value) - len(selected_keys),
        }
    return value


def _schema_keys(items: list[Any]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
            if len(keys) >= 30:
                return keys
    return keys


def _value_score(value: Any) -> int:
    if isinstance(value, dict):
        score = sum(20 for key in value.keys() if _is_important_key(key))
        score += sum(_value_score(item) for item in value.values())
        return score
    if isinstance(value, list):
        return sum(_value_score(item) for item in value[:20])
    if isinstance(value, str):
        lowered = value.lower()
        return sum(10 for keyword in _IMPORTANT_KEYS if keyword in lowered)
    return 0


def _is_important_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _IMPORTANT_KEYS or any(
        marker in lowered for marker in ("error", "fail", "warn", "trace")
    )


__all__ = ["compact_json"]
