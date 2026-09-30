"""web 搜索结果（Tavily）的类型感知压缩器。

web_search 的 output 是 JSON 字符串：``{query, answer, results: [{title, url,
content, score, favicon}], ...}``。压缩策略：按域名去重、保留 Top-K、截断摘要、
丢弃 favicon 噪声。无法解析或无结果时返回 None。
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

_MAX_RESULTS = 8
_MAX_SNIPPET_CHARS = 300
_MAX_PER_DOMAIN = 3


def compact_web_search(
    text: str,
    *,
    max_results: int = _MAX_RESULTS,
    max_snippet_chars: int = _MAX_SNIPPET_CHARS,
    max_per_domain: int = _MAX_PER_DOMAIN,
) -> str | None:
    """压缩 web_search 的 output（JSON 字符串）；无变化时返回 None。"""

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None

    kept: list[dict] = []
    domain_counts: dict[str, int] = {}
    omitted = 0
    changed = False

    for item in results:
        if not isinstance(item, dict):
            kept.append(item)
            continue

        domain = _domain(item.get("url"))
        if len(kept) >= max_results:
            omitted += 1
            changed = True
            continue
        if domain and domain_counts.get(domain, 0) >= max_per_domain:
            omitted += 1
            changed = True
            continue

        content = item.get("content")
        if isinstance(content, str) and len(content) > max_snippet_chars:
            item = {**item, "content": content[:max_snippet_chars] + "…"}
            changed = True

        kept.append(item)
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    if not changed:
        return None

    new_data = {**data, "results": kept}
    new_data["counts"] = len(results)  # 保留原始总数
    new_data["omitted"] = omitted
    return json.dumps(new_data, ensure_ascii=False, separators=(",", ":"))


def _domain(url: object) -> str:
    if not isinstance(url, str) or not url:
        return ""
    netloc = urlsplit(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc


__all__ = ["compact_web_search"]
