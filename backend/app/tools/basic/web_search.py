"""与模型无关的只读网页搜索工具。"""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolDefinition, ToolPermission
from ..search import (
    SearchRequest,
    SearchService,
    SearchSettings,
    build_search_service,
)

MAX_QUERY_CHARS = 500


class WebSearchTool(BaseTool):
    """通过统一 SearchService 执行 Tavily 搜索。"""

    def __init__(
        self,
        *,
        service: SearchService | None = None,
        settings: SearchSettings | None = None,
    ) -> None:
        config = settings or SearchSettings()
        self._service = service or build_search_service(config)
        self._max_results = config.search_max_results

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="搜索网页，返回来源标题、URL、简短摘要及可用的相关性评分。此工具只读，当前需要人工审批。最终回答应引用返回的来源。使用少量聚焦的搜索，避免反复变换宽泛的查询。涉及相对日期时，应先调用 get_current_time 获取当前时间，不要自行假定日期。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "聚焦、明确的搜索词。",
                        "maxLength": MAX_QUERY_CHARS,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回的结果数量。",
                        "default": self._max_results,
                        "minimum": 1,
                        "maximum": self._max_results,
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "default": "general",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "可选的时间范围筛选条件。",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            # 可选参数不满足 OpenAI 严格模式“全部字段必须 required”的约束，
            # 参数正确性由 SearchRequest 在本地统一验证。
            permission=ToolPermission.HUMAN_APPROVAL,
        )

    @property
    def provider_name(self) -> str:
        """返回当前首选搜索提供商名称，供 CLI 显示。"""

        return self._service.primary_provider

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_max_results = arguments.get("max_results", self._max_results)
        if type(raw_max_results) is not int or raw_max_results < 1:
            raise ValueError("max_results must be a positive integer")
        params = {
            **arguments,
            "max_results": min(raw_max_results, self._max_results),
        }
        request = SearchRequest(**params)
         
        response = await self._service.search(request)
        output = response.model_dump(mode="json")
        output["counts"] = len(response.results)
        return output
        


__all__ = ["MAX_QUERY_CHARS", "WebSearchTool"]


def main() -> None:
    import argparse
    import asyncio
    import json

    from ..search import SearchNetworkError, SearchResponseError

    parser = argparse.ArgumentParser(description="使用 Tavily 搜索网页")
    parser.add_argument("query", help="搜索词")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--topic", choices=["general", "news", "finance"], default="general")
    args = parser.parse_args()
    arguments = {"query": args.query, "topic": args.topic}
    if args.max_results is not None:
        arguments["max_results"] = args.max_results
    try:
        output = asyncio.run(WebSearchTool().execute(arguments))
    except (ValueError, SearchNetworkError, SearchResponseError) as exc:
        parser.exit(1, f"搜索失败：{exc}\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
