"""工具结果的内容类型识别与分发（类型感知压缩入口）。

只负责：解包 ToolResult envelope → 取出真正的输出文本 → 识别类型 → 分发到对应
压缩器 → 回写 envelope。具体压缩算法在各 content/*.py 里。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .build_output import compact_build_output
from .git_diff import compact_git_diff
from .http_response import compact_http_response
from .json import compact_json
from .search_results import compact_search_results
from .web_search import compact_web_search

# command 里命中 grep/rg，才走 grep 识别
_GREP_TOOL_HINT = re.compile(r"\b(?:grep|rg)\b")
_RG_HINT = re.compile(r"\brg\b")
# 短 flag 集群（-rn、-c、...），要求前面是空白，避免误命中 pattern 里的 "-c"
_FLAG_CLUSTER = re.compile(r"(?<!\S)-([a-zA-Z]+)")
# 长选项（--count、--files-with-matches、...）
_LONG_FLAG = re.compile(r"--([a-z][a-z-]*)")
# 这些 flag 会让 grep/rg 输出非「行号」格式，本身就很紧凑，跳过压缩
_NON_LINE_SHORT = frozenset("clLo")  # c=count, l/L=files-only, o=only-matching
_NON_LINE_LONG = frozenset(
    {"count", "files-with-matches", "files-without-match", "only-matching"}
)
# grep -n 的多文件形态：file:行号:内容
_SEARCH_RESULT_PATTERN = re.compile(r"^[^\s:][^:\n]*:\d+:", re.MULTILINE)
# git diff / unified diff
_GIT_DIFF_HINT = re.compile(r"\bdiff\b")
_DIFF_HEADER_PATTERN = re.compile(r"^(?:diff --git |@@\s+-\d+)", re.MULTILINE)
# build / pytest / 编译输出
_BUILD_HINT = re.compile(
    r"\b(?:pytest|cargo|go test|npm|cmake|make|gradle|mvn|nosetests|unittest)\b",
    re.IGNORECASE,
)
_BUILD_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|=====|\b\d+\s+(?:passed|failed|skipped|error|errors)\b|\bFAILED\b|npm ERR!",
    re.IGNORECASE,
)


def route_compact(content: str, tool_name: str | None) -> str | None:
    """对一条 tool-result 消息的 content 做类型感知压缩。

    ``content`` 是 ``ToolResult.model_dump_json()`` 的字符串；返回值是压缩后的
    新 content 字符串。无法识别、或压缩后不更小时返回 None（调用方走兜底截断）。
    """

    try:
        envelope = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(envelope, dict):
        return None

    output = envelope.get("output")
    if not isinstance(output, str):
        return None

    if tool_name == "run_shell_command":
        compacted = _compact_shell_output(output)
    elif tool_name == "web_search":
        compacted = compact_web_search(output)
    elif tool_name == "http_request":
        compacted = compact_http_response(output)
    else:
        return None

    if compacted is None or len(compacted) >= len(output):
        return None

    envelope["output"] = compacted
    new_content = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return new_content if len(new_content) < len(content) else None


def _compact_shell_output(output: str) -> str | None:
    """对 run_shell_command 的 output（JSON 字符串）做类型感知压缩。"""

    shell = _unwrap_shell_result(output)
    if shell is None:
        return None
    command, stdout, shell_data = shell

    compacted_stdout = _compact_stdout(command, stdout)
    if compacted_stdout is None or len(compacted_stdout) >= len(stdout):
        return None

    shell_data["stdout"] = compacted_stdout
    return json.dumps(shell_data, ensure_ascii=False, separators=(",", ":"))


def _compact_stdout(command: str, stdout: str) -> str | None:
    """按内容类型尝试压缩 stdout，返回压缩结果或 None。

    先用 command 判类型（避免「grep 结果里含 error 字样被误判成 build」这类歧义），
    command 不明确时再退回 stdout 内容形状猜测。
    """

    # 1) command 明确是 diff
    if _GIT_DIFF_HINT.search(command):
        compacted = compact_git_diff(stdout)
        if compacted is not None:
            return compacted

    # 2) command 明确是 build/test
    if _BUILD_HINT.search(command):
        compacted = compact_build_output(stdout)
        if compacted is not None:
            return compacted

    # 3) command 明确是 grep/rg
    if _GREP_TOOL_HINT.search(command) and _should_compact_grep(command, stdout):
        compacted = compact_search_results(stdout)
        if compacted is not None:
            return compacted

    # 4) command 不明确，退回内容形状（json → diff → build → grep）
    compacted = compact_json(stdout)
    if compacted is not None:
        return compacted
    if _DIFF_HEADER_PATTERN.search(stdout):
        compacted = compact_git_diff(stdout)
        if compacted is not None:
            return compacted
    if _BUILD_PATTERN.search(stdout):
        compacted = compact_build_output(stdout)
        if compacted is not None:
            return compacted
    if _SEARCH_RESULT_PATTERN.search(stdout):
        compacted = compact_search_results(stdout)
        if compacted is not None:
            return compacted

    return None


def _should_compact_grep(command: str, stdout: str) -> bool:
    """判断是否应按「grep 行号结果」压缩 stdout。

    只压缩带行号的输出（``grep -n`` / 多文件 / ``rg`` 默认）；``-c/-l/-L/-o``
    这些非行号变体本身紧凑，跳过。
    """

    if _GREP_TOOL_HINT.search(command):
        shorts: set[str] = set()
        for cluster in _FLAG_CLUSTER.findall(command):
            shorts.update(cluster)
        longs = set(_LONG_FLAG.findall(command))

        if shorts & _NON_LINE_SHORT or longs & _NON_LINE_LONG:
            return False
        # rg 默认就带行号；grep 必须显式 -n 才带行号
        if _RG_HINT.search(command):
            return True
        return "n" in shorts

    # command 没提 grep（如 find | xargs grep），退回 stdout 形状判断
    return bool(_SEARCH_RESULT_PATTERN.search(stdout))


def _unwrap_shell_result(output: str) -> tuple[str, str, dict[str, Any]] | None:
    """解析 run_shell_command 的 output（JSON 字符串），返回 (command, stdout, dict)。"""

    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    command = data.get("command")
    stdout = data.get("stdout")
    if not isinstance(stdout, str):
        return None
    return (command if isinstance(command, str) else ""), stdout, data


__all__ = ["route_compact"]
