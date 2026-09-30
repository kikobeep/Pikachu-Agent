"""git diff / unified diff 输出的类型感知压缩器。

保留增删行（``+``/``-``）与 hunk 头（``@@``），折叠冗长的上下文行（`` `` 开头），
只留少量上下文 + 含 def/class/error 等关键字的上下文。非 diff 输出返回 None。
"""

from __future__ import annotations

_MAX_CONTEXT_LINES = 2
_MAX_FILES = 20

# 上下文行里出现这些词，说明这行值得保留
_IMPORTANT_CONTEXT_KEYWORDS = ("def ", "class ", "function ", "todo", "fixme", "error")


def compact_git_diff(
    text: str,
    *,
    max_context_lines: int = _MAX_CONTEXT_LINES,
    max_files: int = _MAX_FILES,
) -> str | None:
    '''
    读取 diff 文本
        ↓
    检查是否包含 diff 特征
        ↓
    逐行处理
        ├─ 文件头：记录文件并受 max_files 限制
        ├─ @@：保留代码块范围
        ├─ +：保留新增代码
        ├─ -：保留删除代码
        ├─ 空格开头：只保留有限上下文
        └─ 其他行：原样保留
        ↓
    附加省略的上下文行数和文件数
        ↓
    返回压缩后的 diff
    '''

    lines = text.splitlines()
    if not any(
        line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")) for line in lines
    ):
        return None

    compressed: list[str] = ["[Git diff compacted]"]
    files_affected = 0
    context_kept = 0
    context_omitted = 0
    visible_files = 0
    skip_current_file = False
    has_active_file = False

    for line in lines:
        if line.startswith("diff --git "):
            files_affected += 1 # 记录 diff 涉及多少个文件
            has_active_file = True
            context_kept = 0  # 记录当前文件已经保留了多少行普通上下文。
            skip_current_file = visible_files >= max_files
            if skip_current_file:
                continue
            visible_files += 1
            compressed.append("")
            compressed.append(line)
            continue

        if skip_current_file:
            continue

        if line.startswith(("--- ", "+++ ")):
            if not has_active_file:
                # diff -u file1 file2 这类没有 diff --git 头的形态
                files_affected += 1
                has_active_file = True
                context_kept = 0
                skip_current_file = visible_files >= max_files
                if skip_current_file:
                    continue
                visible_files += 1
            compressed.append(line)
            continue

        if line.startswith("@@ "):
            compressed.append(line)
            continue

        if line.startswith("+") and not line.startswith("+++"): # +++ app.py 是文件头，不是新增代码，所以通过
            compressed.append(line)
            continue

        if line.startswith("-") and not line.startswith("---"):
            compressed.append(line)
            continue

        if line.startswith(" "):
            if _is_important_context(line) or context_kept < max_context_lines:
                compressed.append(line)
                context_kept += 1
            else:
                context_omitted += 1
            continue

        # 其它行（如 "\ No newline at end of file"）原样保留
        compressed.append(line)

    hidden_files = max(0, files_affected - visible_files)
    if context_omitted:
        compressed.append(f"[... omitted {context_omitted} context lines]")
    if hidden_files:
        compressed.append(f"[... omitted {hidden_files} diff files]")

    return "\n".join(compressed).strip()


def _is_important_context(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in _IMPORTANT_CONTEXT_KEYWORDS)


__all__ = ["compact_git_diff"]
