"""grep / ripgrep 搜索结果的类型感知压缩器。

把行号结果按文件分组、按「价值」选优，折叠低价值命中，而不是无脑头尾截断。
兼容两种形态：

- 多文件 / ``-H``：``file:行号:内容``
- 单文件 ``grep -n``：``行号:内容``（无文件名前缀）

只压缩、不归档；原始输出仍由 evidence 层保存。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_FILES = 15
_MAX_MATCHES_PER_FILE = 5

# 命中文本里的「高价值」关键词：出现即优先保留
_HIGH_VALUE_KEYWORDS = (
    "error",
    "failed",
    "fail",
    "traceback",
    "exception",
    "todo",
    "fixme",
    "warning",
)
# 全大写常量（SENTINEL、MAX_RETRIES 之类）通常更值得留
_CONSTANT_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{6,}\b")

# 单文件形态：行号:内容
_LINE_NUMBER_PATTERN = re.compile(r"^(\d+):(.*)$")


@dataclass(frozen=True, slots=True)
class _SearchMatch:
    file_path: str
    line_number: int
    text: str


def compact_search_results(
    text: str,
    *,
    max_files: int = _MAX_FILES,
    max_matches_per_file: int = _MAX_MATCHES_PER_FILE,
) -> str | None:
    """
    原始 grep 输出
        ↓
    识别多文件或单文件格式
        ↓
    解析为结构化命中记录
        ↓
    按文件分组
        ↓
    优先保留命中密集的文件
        ↓
    每个文件保留首尾和高价值行
        ↓
    生成带省略统计的紧凑文本
    """

    file_matches = _parse_file_line_matches(text)
    if file_matches:
        return _render_grouped(file_matches, max_files, max_matches_per_file)

    line_matches = _parse_line_number_matches(text)
    if line_matches:
        return _render_single_file(line_matches, max_matches_per_file)

    return None


def _parse_file_line_matches(content: str) -> list[_SearchMatch]:
    """解析多文件形态 ``file:行号:内容``。"""

    matches: list[_SearchMatch] = []
    for line in content.splitlines():
        parsed = _split_search_line(line)
        if parsed is None:
            continue
        file_path, line_number, text = parsed
        matches.append(
            _SearchMatch(file_path=file_path, line_number=line_number, text=text)
        )
    return matches


def _parse_line_number_matches(content: str) -> list[_SearchMatch]:
    """解析单文件形态 ``行号:内容``（无文件名前缀）。"""

    matches: list[_SearchMatch] = []
    for line in content.splitlines():
        match = _LINE_NUMBER_PATTERN.match(line)
        if match is None:
            continue
        matches.append(
            _SearchMatch(
                file_path="",
                line_number=int(match.group(1)),
                text=match.group(2).lstrip(),
            )
        )
    return matches


def _render_grouped(
    matches: list[_SearchMatch],
    max_files: int,
    max_matches_per_file: int,
) -> str:
    grouped: dict[str, list[_SearchMatch]] = {}
    for match in matches:
        grouped.setdefault(match.file_path, []).append(match)

    lines = [
        "[Search results compacted]",
        f"original_matches={len(matches)}",
        f"files={len(grouped)}",
    ]

    # 命中多的文件优先保留（并列再按文件名），避免字母序靠后但命中密集的文件被整体丢弃
    for file_path in sorted(grouped, key=lambda fp: (-len(grouped[fp]), fp))[:max_files]:
        file_matches = sorted(grouped[file_path], key=lambda item: item.line_number)
        selected = _select_search_matches(
            file_matches,
            max_matches=max_matches_per_file,
        )
        omitted = len(file_matches) - len(selected)

        lines.append(f"\n## {file_path} ({len(file_matches)} matches)")
        for match in selected:
            lines.append(f"{match.file_path}:{match.line_number}: {match.text}")
        if omitted:
            lines.append(f"[... omitted {omitted} matches in {file_path}]")

    hidden_files = len(grouped) - max_files
    if hidden_files > 0:
        lines.append(f"\n[... omitted {hidden_files} files]")

    return "\n".join(lines)


def _render_single_file(matches: list[_SearchMatch], max_matches: int) -> str:
    selected = _select_search_matches(matches, max_matches=max_matches)
    omitted = len(matches) - len(selected)

    lines = [
        "[Search results compacted]",
        f"original_matches={len(matches)}",
    ]
    for match in selected:
        lines.append(f"{match.line_number}: {match.text}")
    if omitted:
        lines.append(f"[... omitted {omitted} matches]")

    return "\n".join(lines)


def _split_search_line(line: str) -> tuple[str, int, str] | None:
    """解析 ``path:line:content``；兼容 Windows 盘符（C:\\...）。"""

    first = line.find(":")
    if first == -1:
        return None

    # 处理 "C:\path" 这类 Windows 盘符（第一个冒号出现在位置 1）
    search_from = 3 if first == 1 and len(line) > 2 and line[2] in ("\\", "/") else 0
    second = line.find(":", search_from)
    while second != -1:
        third = line.find(":", second + 1)
        if third == -1:
            return None
        line_number = line[second + 1 : third]
        if line_number.isdigit():
            return line[:second], int(line_number), line[third + 1 :].lstrip()
        second = line.find(":", second + 1)
    return None


def _select_search_matches(
    matches: list[_SearchMatch],
    *,
    max_matches: int,
) -> list[_SearchMatch]:
    """保留首条 + 末条，其余按价值分填充，最后按行号排序返回。"""

    if len(matches) <= max_matches:
        return list(matches)

    selected: list[_SearchMatch] = [matches[0]]
    if max_matches > 1:
        selected.append(matches[-1])

    for match in sorted(matches, key=_search_match_score, reverse=True):
        if len(selected) >= max_matches:
            break
        if match not in selected:
            selected.append(match)

    return sorted(selected, key=lambda item: item.line_number)


def _search_match_score(match: _SearchMatch) -> int:
    text = match.text.lower()
    score = 0
    for keyword in _HIGH_VALUE_KEYWORDS:
        if keyword in text:
            score += 10
    if "sentinel" in text:
        score += 20
    if _CONSTANT_PATTERN.search(match.text):
        score += 12
    score += min(3, len(match.text) // 80)
    return score


__all__ = ["compact_search_results"]
