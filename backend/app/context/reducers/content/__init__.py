"""类型感知工具结果压缩器（content-type-aware tool result compaction）。

每个内容类型一个专用压缩器；router 负责识别类型并分发。压缩只做「留对东西」，
原始字节仍由 evidence 层归档，可随时回读。
"""

from .build_output import compact_build_output
from .git_diff import compact_git_diff
from .http_response import compact_http_response
from .json import compact_json
from .router import route_compact
from .search_results import compact_search_results
from .web_search import compact_web_search

__all__ = [
    "route_compact",
    "compact_search_results",
    "compact_git_diff",
    "compact_build_output",
    "compact_web_search",
    "compact_json",
    "compact_http_response",
]
