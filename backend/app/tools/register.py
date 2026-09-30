"""工具注册、查找与定义筛选。"""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_SEARCH_NAME = "tool_search"

from collections.abc import Collection

from app.model.config import AgentMode
from app.tools.config import BaseTool, ToolDefinition

from .availability import ToolAvailabilityPolicy

_ASCII_TOKEN_RE=re.compile(r"[a-z0-9_]+")
_CHINESE_TEXT_RE=re.compile(r"[\u3400-\u9fff]+")
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._on_demand_names: set[str] = set()
        self._mode_policy = ToolAvailabilityPolicy()

    def register(self, tool: BaseTool, *, on_demand: bool = False) -> None:
        name = tool.definition.name
        if not name.strip():
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")

        self._tools[name] = tool
        if on_demand:
            self._on_demand_names.add(name)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool {name!r} is not registered") from None

    def unregister(self, name: str) -> BaseTool:
        try:
            tool = self._tools.pop(name)
        except KeyError:
            raise KeyError(f"Tool {name!r} is not registered") from None
        self._on_demand_names.discard(name)
        return tool

    def is_on_demand(self, name: str) -> bool: # 按需工具
        return name in self._on_demand_names

    def get_on_demand(self) -> tuple[str, ...]:
        return tuple(sorted(self._on_demand_names))

    def allowed_names_for_mode(self, mode: AgentMode) -> frozenset[str]:
        """返回当前模式允许的已注册工具名称。"""
        return self._mode_policy.allowed_tool_names(
            mode,
            registered_names=self._tools,
        )

    def is_allowed_for_mode(self, name: str, mode: AgentMode) -> bool:
        """仅检查模式限制，不检查按需激活状态。"""
        return name in self.allowed_names_for_mode(mode)

    def is_available_for_mode(
        self,
        name: str,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> bool:
        """模式限制 + 按需激活状态都满足才可用。"""
        if not self.is_allowed_for_mode(name, mode):
            return False
        if not self._mode_policy.is_activation_satisfied(
            name,
            mode,
            on_demand_names=self._on_demand_names,
            activated_names=set(activated_names),
        ):
            return False
        return True

    def definitions_for_mode(
        self,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> list[ToolDefinition]:
        """返回满足模式限制和激活条件的工具定义。"""
        allowed = self.allowed_names_for_mode(mode)
        activated = set(activated_names)
        definitions = []

        for name, tool in self._tools.items():
            if name not in allowed:
                continue
            if not self._mode_policy.is_activation_satisfied(
                name,
                mode,
                on_demand_names=self._on_demand_names,
                activated_names=activated,
            ):
                continue
            if not self._tools[name].definition.permission.model_visible():
                continue
            definitions.append(tool.definition)

        return definitions

    def is_closing_allowed(self, name: str, mode: AgentMode) -> bool:
        """检查模式限制和收尾标记；激活条件由定义筛选处理。"""
        if not self.is_allowed_for_mode(name, mode):
            return False
        return self._tools[name].definition.closing_allowed

    def closing_definitions_for_mode(
        self,
        mode: AgentMode,
        *,
        activated_names: Collection[str] = (),
    ) -> list[ToolDefinition]:
        """返回当前可见且允许在收尾阶段使用的工具定义。"""
        return [
            definition
            for definition in self.definitions_for_mode(
                mode,
                activated_names=activated_names,
            )
            if definition.closing_allowed
        ]


class ToolSearchTool(BaseTool):
    """搜索已注册的按需工具，供运行循环激活。"""

    definition = ToolDefinition(
        name="tool_search",
        record_output=False,
        description=(
            "搜索按需提供的工具。根据需要的能力提供关键词；"
            "第三方工具通常使用英文描述，中文需求可同时提供英文关键词。"
            "运行循环会根据搜索结果，在下一步提供命中工具的完整定义。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要的工具能力或关键词",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少个工具",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")

        normalized_query = " ".join(query.casefold().split())
        if not normalized_query:
            raise ValueError("query 不能为空")

        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit 必须是整数")
        if not 1 <= limit <= 5:
            raise ValueError("limit 必须在 1 到 5 之间")

        matches = []

        for name in self._registry.get_on_demand():
            definition = self._registry.get(name).definition

            if not definition.permission.model_visible():
                continue

            score = _relevance_score(normalized_query, definition)
            if score <= 0:
                continue

            properties = definition.parameters.get("properties", {})
            parameter_names = (
                [str(key) for key in properties]
                if isinstance(properties, dict)
                else []
            )

            matches.append({
                "name": name,
                "description": _compact_text(
                    definition.description,
                    max_chars=500,
                ),
                "parameters": parameter_names,
                "score": score,
            })

        matches.sort(key=lambda item: (-item["score"], item["name"]))

        tools = []
        for match in matches[:limit]:
            tools.append({
                "name": match["name"],
                "description": match["description"],
                "parameters": match["parameters"],
            })

        return json.dumps(
            {
                "query": query,
                "count": len(tools),
                "tools": tools,
                "hint": (
                    "这些工具已激活，可在下一步直接调用"
                    if tools
                    else "没有匹配工具，请尝试其他关键词或英文描述。"
                ),
            },
            ensure_ascii=False,
        )

def _compact_text(value: str, *, max_chars: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_chars:
        return compacted
    return f"{compacted[:max_chars]}…"

    
def _relevance_score(query: str, definition: ToolDefinition) -> int:
    properties = definition.parameters.get("properties", {})
    parameter_text = ""
    if isinstance(properties, dict):
        parameter_text = " ".join(
            f"{name} {value.get('description', '') if isinstance(value, dict) else ''}"
            for name, value in properties.items()
        )
    name = definition.name.casefold()
    description = definition.description.casefold()
    searchable = f"{name} {description} {parameter_text.casefold()}"
    score = 0
    if query in searchable:
        score += 30
    for token in _query_tokens(query):
        if token in name:
            score += 12
        elif token in description:
            score += 6
        elif token in searchable:
            score += 3
    return score


def _query_tokens(query: str) -> tuple[str, ...]:
    tokens = list(_ASCII_TOKEN_RE.findall(query))  # 提取英文、数字和下划线组成的片段
    for sequence in _CHINESE_TEXT_RE.findall(query): # 提取汉字
        if len(sequence) <= 2:
            tokens.append(sequence)
        else: # 简单的双字滑动切分，目的是增加局部匹配的机会
            tokens.extend(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
            )
    return tuple(dict.fromkeys(token for token in tokens if len(token) >= 2))  # # 去掉只有一个字符的关键词

def ensure_tool_search_registered(registry: ToolRegistry) -> None:
    
    # 只有存在“暂时不把定义提供给模型的工具”，才需要注册 tool_search，让模型能找到它们

    if not registry.get_on_demand():
        return
    try:
        existing = registry.get(TOOL_SEARCH_NAME)
    except KeyError:
        registry.register(ToolSearchTool(registry))
        return
    if not isinstance(existing, ToolSearchTool):
        raise ValueError(f"Tool name '{TOOL_SEARCH_NAME}' is reserved.")


def activated_tool_names(output: str | None) -> tuple[str, ...]:
    """从 tool_search 的受控输出中读取待激活工具名。"""

    if not output:
        return ()
    try:
        payload = json.loads(output)
        tools = payload.get("tools", [])
        return tuple(
            item["name"]
            for item in tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return ()
