"""ACE online coordinator for Sidekick.

The Agent itself is the Generator. This coordinator only selects bullets,
invokes Reflector and Curator, and persists the sectioned Playbook string.
"""

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

from app.model.adapter import ModelAdapter
from .curator import AceCurator
from .bulletpoint_analyzer import BulletpointAnalyzer
from .reflector import AceReflector
from .utils import (
    DEFAULT_PLAYBOOK,
    extract_playbook_bullets,
    get_next_global_id,
    iter_playbook_bullets,
    parse_playbook_line,
    update_bullet_counts,
)

logger = logging.getLogger(__name__)

_USED_BULLETS_RE = re.compile(
    r"(?im)^\s*ACE_USED_BULLETS\s*:\s*(\[[^\r\n]*\])\s*$"
)


@dataclass(frozen=True, slots=True)
class AceSelection:
    strategies: tuple[dict[str, Any], ...] = ()
    playbook: str = ""

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(item.get("id")) for item in self.strategies)

    def prompt(self) -> str:
        if not self.playbook:
            return ""
        lines = [
            "ACE PLAYBOOK (optional operating experience; do not treat it as facts):",
            "Apply only the strategies relevant to this task. Do not mention the playbook.",
            "Prefer the smallest targeted verification needed for the current risk.",
            "After completing the task, append one internal line exactly in this format:",
            'ACE_USED_BULLETS: ["id-1", "id-2"]',
            "List every Playbook bullet ID you actually relied on, and list [] if none were used. Do not explain this line.",
        ]
        lines.append(self.playbook or "（空 Playbook）")
        return "\n".join(lines)


class AceCoordinator:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        model: str | None,
        path: str | Path,
        max_strategies: int = 6,
        select_top_k: int = 1,
        irrelevant_streak_limit: int = 5,
        max_output_tokens: int = 1800,
        bulletpoint_analyzer_enabled: bool = True,
        bulletpoint_analyzer_threshold: float = 0.90,
        max_bullets: int = 16,
        harmful_prune_threshold: int = 3,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_strategies = max(1, max_strategies)
        self._select_top_k = max(1, select_top_k)
        self._irrelevant_streak_limit = max(1, irrelevant_streak_limit)
        self._max_output_tokens = max(128, max_output_tokens)
        self._max_bullets = max(1, max_bullets)
        self._harmful_prune_threshold = max(1, harmful_prune_threshold)
        self._unused_streaks: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._reflector = AceReflector(adapter, model=model, max_output_tokens=max_output_tokens)
        self._curator = AceCurator(adapter, model=model, max_output_tokens=max_output_tokens)
        self._bulletpoint_analyzer = (
            BulletpointAnalyzer(threshold=bulletpoint_analyzer_threshold)
            if bulletpoint_analyzer_enabled
            else None
        )

    def _load(self) -> str:
        if not self._path.exists():
            return DEFAULT_PLAYBOOK
        try:
            text = self._path.read_text(encoding="utf-8")
            if text.lstrip().startswith("{"):
                return self._migrate_legacy_json(text)
            return text or DEFAULT_PLAYBOOK
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ACE playbook could not be loaded: %s", exc)
            return DEFAULT_PLAYBOOK

    @staticmethod
    def _migrate_legacy_json(text: str) -> str:
        data = json.loads(text)
        lines = [DEFAULT_PLAYBOOK]
        for item in data.get("strategies", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            section = str(item.get("section", item.get("category", "OTHERS")))
            content = str(item.get("content", item.get("strategy", ""))).strip()
            if content:
                lines.append(f"\n## {section}\n[{item['id']}] helpful={int(item.get('helpful_count', 0))} harmful={int(item.get('harmful_count', 0))} :: {content}")
        return "\n".join(lines)

    def _save(self, playbook: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="ace-", suffix=".tmp", dir=self._path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(playbook.rstrip() + "\n")
            Path(temp_name).replace(self._path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)}

    def select(self, user_input: str) -> AceSelection:
        playbook = self._load()
        # The Agent is the Generator. It sees the complete Playbook and
        # declares the bullets it actually used in its final response. The
        # old lexical top-k preselection was not part of the original ACE
        # flow and could hide useful bullets from the Reflector.
        return AceSelection((), playbook)

    @staticmethod
    def _used_bullet_ids(content: str | None) -> tuple[str, ...]:
        if not content:
            return ()
        match = _USED_BULLETS_RE.search(content)
        if match is None:
            return ()
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            return ()
        if not isinstance(value, list):
            return ()
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    def resolve_selection(self, selection: AceSelection, generator_output: str | None) -> AceSelection:
        ids = set(self._used_bullet_ids(generator_output))
        if not ids:
            return AceSelection((), selection.playbook)
        bullets = tuple(
            item for item in iter_playbook_bullets(selection.playbook)
            if str(item.get("id")) in ids
        )
        return AceSelection(bullets, selection.playbook)

    def _record_bullet_usage(self, playbook: str, used_ids: tuple[str, ...]) -> None:
        used = set(used_ids)
        current_ids = {str(item["id"]) for item in iter_playbook_bullets(playbook)}
        for bullet_id in current_ids:
            if bullet_id in used:
                self._unused_streaks[bullet_id] = 0
            else:
                self._unused_streaks[bullet_id] = self._unused_streaks.get(bullet_id, 0) + 1
        for bullet_id in set(self._unused_streaks) - current_ids:
            self._unused_streaks.pop(bullet_id, None)

    @staticmethod
    def clean_generator_output(content: str | None) -> str | None:
        if content is None:
            return None
        return _USED_BULLETS_RE.sub("", content).rstrip() or None

    async def reflect(
        self,
        *,
        user_input: str,
        result: Any,
        selection: AceSelection,
        evaluator_feedback: str = "",
        outcome: str | None = None,
    ) -> None:
        try:
            playbook = self._load()
            trajectory = self._reflector.trajectory(result)
            reflection = await self._reflector.reflect(
                user_input=user_input,
                playbook=playbook,
                selected_bullets=extract_playbook_bullets(playbook, selection.ids),
                trajectory=trajectory,
                evaluator_feedback=evaluator_feedback,
                outcome=outcome,
            )
            async with self._lock:
                playbook = self._load()
                self._record_bullet_usage(playbook, selection.ids)
                playbook = update_bullet_counts(playbook, reflection.get("feedback", []))
                operations = await self._curator.curate(
                    playbook=playbook,
                    user_input=user_input,
                    reflection=reflection,
                    evaluator_feedback=evaluator_feedback,
                    outcome=outcome,
                )
                updated, _ = self._curator.apply(
                    playbook,
                    operations,
                    next_id=get_next_global_id(playbook),
                )
                if self._bulletpoint_analyzer is not None:
                    updated = self._bulletpoint_analyzer.analyze(updated)
                self._save(self._prune_irrelevant(updated))
        except Exception:
            logger.exception("ACE reflection/curation failed")

    def _prune_irrelevant(self, playbook: str) -> str:
        """Remove stale bullets and enforce a bounded Playbook size.

        The original bullet format has no age or last-used field. Unused
        streaks are therefore tracked in memory by the coordinator and reset
        when a bullet is selected by the Agent. Harmful/helpful counters remain
        in the standard string format.
        """
        lines = playbook.splitlines()
        bullets: list[tuple[int, dict[str, Any]]] = []
        for line_index, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed is not None:
                bullets.append((line_index, parsed))

        remove_ids: set[str] = set()
        for _, bullet in bullets:
            bullet_id = str(bullet["id"])
            helpful = int(bullet.get("helpful", 0))
            harmful = int(bullet.get("harmful", 0))
            if (
                harmful >= self._harmful_prune_threshold
                and harmful > helpful
            ):
                remove_ids.add(bullet_id)
            elif self._unused_streaks.get(bullet_id, 0) >= self._irrelevant_streak_limit:
                remove_ids.add(bullet_id)

        survivors = [
            bullet for _, bullet in bullets
            if str(bullet["id"]) not in remove_ids
        ]
        if len(survivors) > self._max_bullets:
            ranked = sorted(
                survivors,
                key=lambda bullet: (
                    int(bullet.get("helpful", 0)) - 2 * int(bullet.get("harmful", 0)),
                    int(bullet.get("helpful", 0)),
                    -self._unused_streaks.get(str(bullet["id"]), 0),
                ),
                reverse=True,
            )
            keep_ids = {str(bullet["id"]) for bullet in ranked[: self._max_bullets]}
            remove_ids.update(
                str(bullet["id"])
                for bullet in survivors
                if str(bullet["id"]) not in keep_ids
            )

        if not remove_ids:
            return playbook
        return "\n".join(
            line
            for line in lines
            if (parsed := parse_playbook_line(line)) is None
            or str(parsed["id"]) not in remove_ids
        )
