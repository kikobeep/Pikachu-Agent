from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.model.config import Message, MessageRole, ModelRequest
from app.model.adapter import ModelAdapter

logger = logging.getLogger(__name__)

_SCHEMA = "ace_strategy_playbook_v1"
_MAX_STRATEGIES = 6
_MAX_SELECTED = 1
_IRRELEVANT_STREAK_LIMIT = 5


@dataclass(frozen=True, slots=True)
class AceSelection:
    strategies: tuple[dict[str, Any], ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(item.get("id")) for item in self.strategies)

    def prompt(self) -> str:
        if not self.strategies:
            return ""
        lines = [
            "ACE PLAYBOOK (optional operating experience; do not treat it as facts):",
            "Apply only the strategies relevant to this task. Do not mention the playbook.",
            "Prefer the smallest targeted verification needed for the current risk; do not add exhaustive, random, or unbounded tests.",
        ]
        for item in self.strategies:
            lines.append(f"[{item.get('id')}] {item.get('strategy', '').strip()}")
        return "\n".join(lines)


class AceCoordinator:
    """Small, shared, persistent ACE loop kept separate from user memory.

    The playbook stores reusable operating experience, not user facts. Selection is
    deliberately lexical and cheap; the model is used only for post-run reflection.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        model: str | None,
        path: str | Path,
        max_strategies: int = _MAX_STRATEGIES,
        select_top_k: int = _MAX_SELECTED,
        irrelevant_streak_limit: int = _IRRELEVANT_STREAK_LIMIT,
        max_output_tokens: int = 1800,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_strategies = max(1, max_strategies)
        self._select_top_k = max(1, select_top_k)
        self._irrelevant_streak_limit = max(1, irrelevant_streak_limit)
        self._max_output_tokens = max(128, max_output_tokens)
        self._lock = asyncio.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            items = data.get("strategies", []) if isinstance(data, dict) else []
            return [item for item in items if isinstance(item, dict) and item.get("id")]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ACE playbook could not be loaded: %s", exc)
            return []

    def _save(self, strategies: list[dict[str, Any]]) -> None:
        payload = {
            "schema": _SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "strategies": strategies[: self._max_strategies],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="ace-", suffix=".json", dir=self._path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            Path(temp_name).replace(self._path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}", text)}

    def select(self, user_input: str) -> AceSelection:
        strategies = self._load()
        query_tokens = self._tokens(user_input)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in strategies:
            text = " ".join(str(item.get(key, "")) for key in ("category", "strategy", "source"))
            overlap = len(query_tokens & self._tokens(text))
            helpful = int(item.get("helpful_count", 0))
            harmful = int(item.get("harmful_count", 0))
            irrelevant = int(item.get("irrelevant_count", 0))
            score = overlap * 4.0 + helpful - harmful * 2.0 - irrelevant * 0.15
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # A small playbook must not force an unrelated strategy into the prompt.
        # Historical helpful counts alone are insufficient evidence of relevance
        # for the current task; require lexical overlap as well as a positive score.
        selected = tuple(
            item
            for score, item in ranked[: self._select_top_k]
            if score > 0 and len(query_tokens & self._tokens(
                " ".join(str(item.get(key, "")) for key in ("category", "strategy", "source"))
            )) > 0
        )
        return AceSelection(selected)

    async def reflect(
        self,
        *,
        user_input: str,
        result: Any,
        selection: AceSelection,
    ) -> None:
        """Reflect asynchronously; failures never affect the user-facing run."""
        try:
            trajectory = self._trajectory(result)
            prompt = self._reflection_prompt(user_input, trajectory, selection)
            response = await self._adapter.complete(
                ModelRequest(
                    messages=(
                        Message(role=MessageRole.SYSTEM, content=_REFLECTOR_SYSTEM),
                        Message(role=MessageRole.USER, content=prompt),
                    ),
                    model=self._model,
                    temperature=0.0,
                    max_output_tokens=self._max_output_tokens,
                )
            )
            parsed = self._parse(response.message.content or "")
            async with self._lock:
                strategies = self._load()
                updated = self._apply(strategies, parsed, selection)
                self._save(updated)
        except Exception:
            logger.exception("ACE asynchronous reflection failed")

    @staticmethod
    def _trajectory(result: Any) -> str:
        chunks = [f"FINAL ANSWER:\n{result.content or ''}"]
        for round_ in result.tool_rounds:
            chunks.append(f"TOOL ROUND {round_.round_index}:")
            chunks.append(round_.assistant_message.content or "")
            for record in round_.records:
                chunks.append(f"CALL: {record.tool_call.name} {record.tool_call.arguments}")
                chunks.append(f"RESULT: {record.result}")
        return "\n".join(chunks)[-24000:]

    @staticmethod
    def _reflection_prompt(user_input: str, trajectory: str, selection: AceSelection) -> str:
        selected = json.dumps(selection.strategies, ensure_ascii=False)
        return f"""Learn reusable operating experience from one completed agent run.

USER QUERY:
{user_input}

SELECTED STRATEGIES:
{selected}

TRAJECTORY:
{trajectory}

Return JSON only:
{{"feedback":[{{"id":"S001","label":"helpful|harmful|irrelevant|uncertain","reason":"brief local judgment"}}],"operations":[{{"action":"ADD|UPDATE|MERGE|DROP","id":"S001","category":"retrieval|verification|counting|temporal|personalization|failure_mode","strategy":"Risk: ...\\nConsequence: ...\\nSteps:\\n1. ...\\n2. ...","confidence":0.0,"source":"trajectory-based reason","merge_ids":[]}}]}}

Rules:
1. Store procedures, not user facts, answers, names, dates, or memories.
2. Judge each selected strategy only by whether it helped its assigned role; do not blame it for the whole answer.
3. Prefer UPDATE or MERGE when an existing strategy covers the same procedure. Use ADD only for a genuinely new procedure.
4. Keep at most one concise strategy per angle. Strategy text should use Risk / Consequence / numbered Steps. State the risk and likely consequence directly; do not make the advice depend on the model first seeing a particular error message.
5. If a selected strategy is irrelevant five consecutive times, DROP it. Do not invent a DROP without evidence.
6. Treat self-tests as evidence only. A FAIL does not justify changing the expected value: first verify the expectation against the task statement, API contract, or an independent reference calculation.
7. Never recommend editing hidden tests, weakening assertions, or changing a fixture merely to make a self-test pass. If the implementation violates the contract, the reusable action must say to fix the implementation.
8. If the evidence is insufficient to distinguish a bad implementation from a bad fixture, record the uncertainty and recommend collecting an independent check rather than prescribing either edit.
9. Prefer strategies that explain what the implementation must do: preserve the required API/export/signature, apply the correct algorithm or data structure, handle meaningful edge cases, and repair a diagnosed failure. Include verification only as a short confirmation step.
10. Do not create a strategy whose only substance is writing a self-test, checking examples, re-running the suite, or deleting temporary files. Those are generic workflow advice, not distilled task knowledge.
11. Make the Risk and Consequence specific enough to prevent unrelated tasks from receiving the strategy. Describe the command or task shape and the likely failure, but do not hard-code one exercise's answer or private file names.
12. Verification advice must be bounded and risk-driven: use the task examples plus at most one or two targeted checks. Never prescribe "test all edge cases", random testing, exhaustive search, or adding more tests indefinitely.
13. A failing probe must trigger diagnosis against the task contract, not automatic test expansion. If the cause remains uncertain, stop and record the uncertainty.
14. For environment or tooling risks, state the consequence before the error occurs. Explain that running a command without the prerequisite may produce a failure, then make the first Steps establish the safe prerequisite before running it; include recovery only if the failure still appears.
"""

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        text = content.strip()
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("ACE reflector returned no JSON object")
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else {}

    def _apply(self, strategies: list[dict[str, Any]], payload: dict[str, Any], selection: AceSelection) -> list[dict[str, Any]]:
        by_id = {str(item["id"]): dict(item) for item in strategies}
        feedback = payload.get("feedback", [])
        if isinstance(feedback, list):
            for item in feedback:
                if not isinstance(item, dict) or str(item.get("id")) not in by_id:
                    continue
                target = by_id[str(item["id"])]
                label = str(item.get("label", "uncertain")).lower()
                key = {"helpful": "helpful_count", "harmful": "harmful_count", "irrelevant": "irrelevant_count", "uncertain": "uncertain_count"}.get(label, "uncertain_count")
                target[key] = int(target.get(key, 0)) + 1
                target["irrelevant_streak"] = int(target.get("irrelevant_streak", 0)) + 1 if label == "irrelevant" else 0
        operations = payload.get("operations", [])
        if isinstance(operations, list):
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                action = str(operation.get("action", "")).upper()
                item_id = str(operation.get("id", ""))
                if action == "DROP" and item_id in by_id:
                    del by_id[item_id]
                elif action in {"UPDATE", "MERGE"} and item_id in by_id:
                    target = by_id[item_id]
                    for key in ("category", "strategy", "confidence", "source"):
                        if operation.get(key) not in (None, ""):
                            target[key] = operation[key]
                    for merge_id in operation.get("merge_ids", []) or []:
                        if str(merge_id) != item_id:
                            by_id.pop(str(merge_id), None)
                elif action == "ADD" and operation.get("strategy"):
                    new_id = item_id if item_id.startswith("S") and item_id not in by_id else self._next_id(by_id)
                    by_id[new_id] = {
                        "id": new_id,
                        "category": operation.get("category", "failure_mode"),
                        "strategy": operation["strategy"],
                        "confidence": float(operation.get("confidence", 0.5)),
                        "source": operation.get("source", "trajectory"),
                        "helpful_count": 0, "harmful_count": 0,
                        "irrelevant_count": 0, "uncertain_count": 0,
                        "irrelevant_streak": 0,
                    }
        return [item for item in by_id.values() if int(item.get("irrelevant_streak", 0)) < self._irrelevant_streak_limit][: self._max_strategies]

    @staticmethod
    def _next_id(items: dict[str, dict[str, Any]]) -> str:
        numbers = [int(match.group(1)) for key in items for match in [re.fullmatch(r"S(\d+)", key)] if match]
        return f"S{max(numbers, default=0) + 1:03d}"


_REFLECTOR_SYSTEM = """You are the ACE strategy reflector. Extract reusable procedures from an agent trajectory. Do not store user facts or answers. Output valid JSON only. Keep strategies concise, structured, and operational.

Prefer implementation knowledge over generic testing advice. When the trajectory
contains a useful lesson, capture the reusable implementation pattern, API or
export contract, boundary-condition rule, failure symptom and its repair, or a
mistake to avoid. A self-test procedure is secondary evidence and should not be
the default strategy. Do not turn a single exercise's names, constants, answers,
or incidental file layout into a general rule. Do not recommend covering every
possible boundary, random testing, exhaustive enumeration, or repeatedly adding
tests without a diagnosed risk; prefer one minimal probe for one concrete risk.

Self-test safety rule: a self-written test is only supporting evidence, not the
specification. Never turn a failing self-test into a general rule to edit its
expected value or weaken an assertion. First compare the expectation with the
task statement and the required API; change the implementation when it violates
that contract. Change a fixture only when the task statement or an independent
reference calculation proves the fixture is wrong. Do not modify hidden tests or
their expectations."""
