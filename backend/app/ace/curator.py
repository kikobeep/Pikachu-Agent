"""ACE Curator: propose and validate deltas against a string Playbook."""

from __future__ import annotations

import json
import re
from typing import Any

from app.model.adapter import ModelAdapter
from app.model.config import Message, MessageRole, ModelRequest
from .utils import apply_curator_operations


_CURATOR_SYSTEM = """You are the ACE Curator. Update a sectioned ACE Playbook from one Reflector diagnosis.

Return JSON only:
{"operations":[{"action":"ADD|UPDATE|MERGE|DROP","id":"str-00001","section":"FORMULAS & CALCULATIONS","content":"reusable procedure","merge_ids":[]}]}

Rules:
1. The Playbook is a string organized by ## sections. Use section/content, not category/strategy.
2. Output deltas only. Never rewrite the complete Playbook.
3. Review every existing bullet before proposing an ADD. Drop proposals that are already covered by the Playbook, duplicate an existing bullet, are vague, are merely restatements, or have low durable value.
4. ADD only a genuinely new, reusable procedure. Do not add user facts, answers, names, dates, hidden-test details, or private file names.
5. Prefer UPDATE when an existing bullet is useful but incomplete or inaccurate. Prefer MERGE when multiple bullets teach substantially overlapping behavior; preserve the strongest combined content and list the source IDs in merge_ids.
6. Use DROP only for an obsolete, harmful, or clearly incorrect bullet, and require concrete evidence.
7. Be conservative: a failed run does not automatically justify a new bullet. If the Playbook already contains the lesson, return no operation.
8. If no durable update is justified, return {"operations":[]}.
"""


class AceCurator:
    def __init__(self, adapter: ModelAdapter, *, model: str | None, max_output_tokens: int) -> None:
        self._adapter = adapter
        self._model = model
        self._max_output_tokens = max(128, max_output_tokens)

    async def curate(
        self,
        *,
        playbook: str,
        user_input: str,
        reflection: dict[str, Any],
        evaluator_feedback: str,
        outcome: str | None,
    ) -> list[dict[str, Any]]:
        prompt = json.dumps(
            {
                "user_input": user_input,
                "outcome": outcome or "unknown",
                "evaluator_feedback": evaluator_feedback or "",
                "reflection": reflection,
                "current_playbook": playbook,
            },
            ensure_ascii=False,
            indent=2,
        )
        response = await self._adapter.complete(
            ModelRequest(
                messages=(
                    Message(role=MessageRole.SYSTEM, content=_CURATOR_SYSTEM),
                    Message(role=MessageRole.USER, content=prompt),
                ),
                model=self._model,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens,
            )
        )
        return self._parse_operations(response.message.content or "")

    @staticmethod
    def _parse_operations(content: str) -> list[dict[str, Any]]:
        text = content.strip()
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("curator returned no JSON object")
        payload = json.loads(text[start : end + 1])
        raw = payload.get("operations", []) if isinstance(payload, dict) else []
        if not isinstance(raw, list):
            raise ValueError("curator operations must be a list")
        valid: list[dict[str, Any]] = []
        for raw_item in raw:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            action = str(item.get("action", "")).upper()
            item["action"] = action
            if action not in {"ADD", "UPDATE", "MERGE", "DROP"}:
                continue
            if action in {"UPDATE", "MERGE", "DROP"} and not str(item.get("id", "")):
                continue
            if action in {"ADD", "UPDATE", "MERGE"} and not str(item.get("content", "")).strip():
                continue
            if action == "MERGE" and not isinstance(item.get("merge_ids", []), list):
                continue
            item["id"] = str(item.get("id", ""))
            item["section"] = str(item.get("section", "OTHERS"))
            item["content"] = str(item.get("content", "")).strip()
            valid.append(item)
        return valid

    @staticmethod
    def apply(playbook: str, operations: list[dict[str, Any]], *, next_id: int) -> tuple[str, int]:
        return apply_curator_operations(playbook, operations, next_id)
