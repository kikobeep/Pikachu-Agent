"""ACE Reflector: diagnose a completed Agent run without mutating the Playbook."""

from __future__ import annotations

import json
import re
from typing import Any

from app.model.adapter import ModelAdapter
from app.model.config import Message, MessageRole, ModelRequest


_REFLECTOR_SYSTEM = """You are the ACE Reflector. Diagnose one completed agent run and extract reusable lessons.

Return JSON only:
{"reflection":"diagnosis and reusable lesson","feedback":[{"id":"str-00001","label":"helpful|harmful|neutral","reason":"brief local judgment"}],"insights":["optional candidate lessons for the Curator"]}

Rules:
1. Store reusable procedures, not user facts, answers, names, dates, hidden-test details, or private file names.
2. When external evaluator feedback is present, treat passed/failed and concrete test output as authoritative.
3. Judge each selected bullet only by whether it helped this run; do not blame it for the whole result.
4. Prefer specific implementation lessons over generic advice such as "write more tests".
5. Keep insights concise and actionable. The separate Curator decides whether they become Playbook deltas.
"""


class AceReflector:
    def __init__(self, adapter: ModelAdapter, *, model: str | None, max_output_tokens: int) -> None:
        self._adapter = adapter
        self._model = model
        self._max_output_tokens = max(128, max_output_tokens)

    async def reflect(
        self,
        *,
        user_input: str,
        playbook: str,
        selected_bullets: str,
        trajectory: str,
        evaluator_feedback: str = "",
        outcome: str | None = None,
    ) -> dict[str, Any]:
        # Match original ACE: the Reflector judges only the bullets declared
        # as used by the Generator. The Curator, not the Reflector, receives
        # the complete Playbook.
        prompt = f"""USER QUERY:\n{user_input}\n\nSELECTED BULLETS USED BY GENERATOR:\n{selected_bullets or '（无）'}\n\nEXTERNAL EVALUATOR OUTCOME:\n{outcome or 'unknown'}\n\nEXTERNAL EVALUATOR FEEDBACK:\n{evaluator_feedback or '（无）'}\n\nTRAJECTORY:\n{trajectory}\n"""
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
        return self._parse(response.message.content or "")

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        text = content.strip()
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("reflector returned no JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("reflector response must be an object")
        feedback = value.get("feedback", [])
        value["feedback"] = feedback if isinstance(feedback, list) else []
        value["insights"] = value.get("insights", []) if isinstance(value.get("insights", []), list) else []
        return value

    @staticmethod
    def trajectory(result: Any) -> str:
        chunks = [f"FINAL ANSWER:\n{result.content or ''}"]
        for round_ in result.tool_rounds:
            chunks.append(f"TOOL ROUND {round_.round_index}:")
            chunks.append(round_.assistant_message.content or "")
            for record in round_.records:
                chunks.append(f"CALL: {record.tool_call.name} {record.tool_call.arguments}")
                chunks.append(f"RESULT: {record.result}")
        return "\n".join(chunks)[-24000:]
