"""build_output / pytest / 编译报错的类型感知压缩器。

保留报错块（错误行 + 上下文 + 后续堆栈）、摘要行（``===`` 分隔、``N passed/failed``），
折叠中间的无价值日志（大量通过用例、普通打印）。非 build 输出返回 None。
"""

from __future__ import annotations

import re

_MAX_ERROR_BLOCKS = 8
_CONTEXT_LINES = 2
_MAX_WARNINGS = 6
_MAX_SUMMARY_LINES = 12
_MAX_TOTAL_LINES = 120

_ERROR_RE = re.compile(
    r"(ERROR|FAILED|FAIL\b|FATAL|Traceback \(most recent call last\)|Exception|AssertionError|npm ERR!|panic|error:)",
    re.IGNORECASE,
)
_WARNING_RE = re.compile(r"\b(WARN|WARNING|warning:)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(
    r"(^=+|^-+|^_{3,}.*_{3,}$|\b\d+\s+(passed|failed|skipped|error|errors|warning|warnings)\b|"
    # 第三个分支：Build/Compile/Test 后紧跟着结果词（succeeded/failed/...），
    # 用 [^:\n] 限制在同一行内且不跨冒号，避免把 pytest 的 "TestX::test_0 PASSED"
    # 这种测试结果行误判成摘要行。
    r"\b(Build|Compile|Tests?|Suites?)\b[^:\n]{0,40}(succeeded|failed|complete|passed)\b)",
    re.IGNORECASE,
)
_STACK_RE = re.compile(
    r"(^\s*File \".+\", line \d+|^\s*at .+\(.+:\d+:\d+\)|^\s+at [\w.$]+|^\s*-->\s+.+:\d+:\d+)",
    re.IGNORECASE,
)
# pytest 失败头：____ 测试名 ____
_PYTEST_HEADER_RE = re.compile(r"^_{3,}.*_{3,}$")


def compact_build_output(
    text: str,
    *,
    max_error_blocks: int = _MAX_ERROR_BLOCKS,
    context_lines: int = _CONTEXT_LINES,
    max_warnings: int = _MAX_WARNINGS,
    max_summary_lines: int = _MAX_SUMMARY_LINES,
    max_total_lines: int = _MAX_TOTAL_LINES,
) -> str | None:
    '''
    读取构建 / pytest 输出
        ↓
    判断是否像错误日志
        ↓
    找出错误、警告、摘要、堆栈
        ↓
    错误附近保留上下文
        ↓
    错误后保留 JavaScript / Java 等堆栈
        ↓
    错误前保留 pytest traceback
        ↓
    保留 pytest 失败标题和最终摘要
        ↓
    超过总行数时按重要程度裁剪
        ↓
    输出压缩结果和省略统计
    '''

    lines = text.splitlines()
    if not lines or not _looks_like_build_output(lines):
        return None

    selected = _select_indexes(
        lines,
        max_error_blocks=max_error_blocks,
        context_lines=context_lines,
        max_warnings=max_warnings,
        max_summary_lines=max_summary_lines,
        max_total_lines=max_total_lines,
    )
    if not selected:
        return None

    output: list[str] = ["[Build output compacted]"]
    last_index: int | None = None
    for index in sorted(selected):
        if last_index is not None and index > last_index + 1:
            output.append(f"[... omitted {index - last_index - 1} lines]")
        output.append(lines[index])
        last_index = index

    omitted_lines = max(0, len(lines) - len(selected))
    if omitted_lines:
        output.append(f"[... omitted {omitted_lines} total lines]")

    return "\n".join(output)


def _looks_like_build_output(lines: list[str]) -> bool:
    joined = "\n".join(lines)
    return bool(
        _ERROR_RE.search(joined)
        or _WARNING_RE.search(joined)
        or _SUMMARY_RE.search(joined)
    )


def _select_indexes(
    lines: list[str],
    *,
    max_error_blocks: int,
    context_lines: int,
    max_warnings: int,
    max_summary_lines: int,
    max_total_lines: int,
) -> set[int]:
    selected: set[int] = set()

    error_indexes = [i for i, line in enumerate(lines) if _ERROR_RE.search(line)]
    warning_indexes = [i for i, line in enumerate(lines) if _WARNING_RE.search(line)]
    summary_indexes = [i for i, line in enumerate(lines) if _SUMMARY_RE.search(line)]
    stack_indexes = [i for i, line in enumerate(lines) if _STACK_RE.search(line)]

    for index in _first_last(error_indexes, max_error_blocks):
        _add_context(selected, index, line_count=len(lines), radius=context_lines)
        _add_following_stack(selected, lines, index)
        _add_preceding_traceback(selected, lines, index)

    for index in warning_indexes[:max_warnings]:
        _add_context(selected, index, line_count=len(lines), radius=1)

    for index in summary_indexes[:max_summary_lines]:
        selected.add(index)

    for index in stack_indexes:
        selected.add(index)

    if selected:
        selected.add(0)
        selected.add(len(lines) - 1)

    if len(selected) > max_total_lines:
        priority = _rank_indexes(lines, selected)
        return set(sorted(priority[:max_total_lines]))
    return selected


def _first_last(indexes: list[int], max_count: int) -> list[int]:
    if len(indexes) <= max_count:
        return list(indexes)
    selected = [indexes[0]]
    if max_count > 1:
        selected.append(indexes[-1])
    for index in indexes[1:-1]:
        if len(selected) >= max_count:
            break
        selected.append(index)
    return sorted(set(selected))


def _add_context(selected: set[int], index: int, *, line_count: int, radius: int) -> None:
    for item in range(max(0, index - radius), min(line_count, index + radius + 1)):
        selected.add(item)


def _add_following_stack(selected: set[int], lines: list[str], index: int) -> None:
    for next_index in range(index + 1, min(len(lines), index + 12)):
        line = lines[next_index]
        if _STACK_RE.search(line) or line.startswith((" ", "\t")):
            selected.add(next_index)
            continue
        break


def _add_preceding_traceback(selected: set[int], lines: list[str], index: int) -> None:
    """从错误行向前捕 pytest traceback（def / > / 缩进代码），直到非 traceback 边界。

    pytest 的 traceback 在错误行（E AssertionError）之前，而不是之后；Java/JS 的
    堆栈在错误行之后。这里补上「向前」那一半。
    """

    for prev_index in range(index - 1, max(-1, index - 13), -1):
        line = lines[prev_index]
        if _PYTEST_HEADER_RE.match(line):
            selected.add(prev_index)
            break
        stripped = line.strip()
        if not stripped:
            selected.add(prev_index)
            continue
        if line.startswith((" ", "\t")) or stripped.startswith((">", "def ", "async def ")):
            selected.add(prev_index)
            continue
        break


def _rank_indexes(lines: list[str], indexes: set[int]) -> list[int]:
    return sorted(indexes, key=lambda index: (_line_score(lines[index]), -index), reverse=True)


def _line_score(line: str) -> int:
    if _ERROR_RE.search(line):
        return 100
    if _STACK_RE.search(line):
        return 90
    if _SUMMARY_RE.search(line):
        return 70
    if _WARNING_RE.search(line):
        return 50
    return 10


__all__ = ["compact_build_output"]
