"""Post-curation duplicate detection for the ACE string Playbook.

The original ACE implementation uses sentence-transformer embeddings and an
LLM to merge highly similar bullets. Sidekick keeps that behavior dependency
free: it uses an optional embedding backend when available and otherwise falls
back to a conservative lexical similarity check. Merging is deterministic so
the analyzer cannot add another unreliable model call to the evaluation path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .utils import (
    format_playbook_line,
    parse_playbook_line,
)


_TOKEN_RE = re.compile(r"[^\W_]{2,}", flags=re.UNICODE)
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class _BulletLocation:
    line_index: int
    bullet: dict[str, Any]


class BulletpointAnalyzer:
    """Find and merge near-duplicate bullets after Curator operations."""

    def __init__(
        self,
        *,
        threshold: float = 0.90,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = min(1.0, max(0.5, float(threshold)))
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None

    @staticmethod
    def _tokens(text: str) -> Counter[str]:
        return Counter(token.lower() for token in _TOKEN_RE.findall(text))

    @classmethod
    def _lexical_similarity(cls, left: str, right: str) -> float:
        left_counts = cls._tokens(left)
        right_counts = cls._tokens(right)
        if not left_counts or not right_counts:
            return 0.0
        common = sum((left_counts & right_counts).values())
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        return common / (left_norm * right_norm)

    def _embedding_similarity(self, left: str, right: str) -> float | None:
        """Return cosine similarity if sentence-transformers is installed."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                return None
            try:
                self._embedding_model = _EMBEDDING_MODEL_CACHE.get(self.embedding_model_name)
                if self._embedding_model is None:
                    self._embedding_model = SentenceTransformer(
                        self.embedding_model_name,
                        local_files_only=True,
                    )
                    _EMBEDDING_MODEL_CACHE[self.embedding_model_name] = self._embedding_model
            except Exception:
                return None
        try:
            vectors = self._embedding_model.encode([left, right], normalize_embeddings=True)
            return float(vectors[0] @ vectors[1])
        except Exception:
            return None

    def _similarity(self, left: str, right: str) -> float:
        semantic = self._embedding_similarity(left, right)
        if semantic is not None:
            return semantic
        # The original threshold is calibrated for embeddings. A lexical
        # cosine score is lower for harmless wording changes, so use a small
        # fallback margin while keeping the default conservative.
        return self._lexical_similarity(left, right)

    def _is_duplicate(self, left: str, right: str) -> bool:
        score = self._similarity(left, right)
        if self._embedding_model is not None:
            return score >= self.threshold
        return score >= max(0.75, self.threshold - 0.15)

    @staticmethod
    def _merge_group(group: list[_BulletLocation]) -> dict[str, Any]:
        # Keep the most useful existing wording and combine its evidence
        # counters. This preserves the original ACE invariant that IDs remain
        # stable after a merge.
        target = max(
            group,
            key=lambda item: (
                int(item.bullet.get("helpful", 0)) - int(item.bullet.get("harmful", 0)),
                len(str(item.bullet.get("content", ""))),
            ),
        ).bullet
        return {
            "id": target["id"],
            "helpful": sum(int(item.bullet.get("helpful", 0)) for item in group),
            "harmful": sum(int(item.bullet.get("harmful", 0)) for item in group),
            "content": target.get("content", ""),
        }

    def analyze(self, playbook: str) -> str:
        lines = playbook.splitlines()
        locations: list[_BulletLocation] = []
        for line_index, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed is not None:
                locations.append(_BulletLocation(line_index, parsed))
        if len(locations) < 2:
            return playbook

        groups: list[list[_BulletLocation]] = []
        consumed: set[str] = set()
        for index, location in enumerate(locations):
            bullet_id = str(location.bullet["id"])
            if bullet_id in consumed:
                continue
            group = [location]
            for other in locations[index + 1 :]:
                other_id = str(other.bullet["id"])
                if other_id in consumed:
                    continue
                if self._is_duplicate(
                    str(location.bullet.get("content", "")),
                    str(other.bullet.get("content", "")),
                ):
                    group.append(other)
            if len(group) > 1:
                groups.append(group)
                consumed.update(str(item.bullet["id"]) for item in group)

        if not groups:
            return playbook

        replacements = {
            str(group[0].bullet["id"]): self._merge_group(group)
            for group in groups
        }
        removed = {
            str(item.bullet["id"])
            for group in groups
            for item in group[1:]
        }

        output: list[str] = []
        for line in lines:
            parsed = parse_playbook_line(line)
            if parsed is None:
                output.append(line)
                continue
            bullet_id = str(parsed["id"])
            if bullet_id in removed:
                continue
            merged = replacements.get(bullet_id)
            if merged is None:
                output.append(line)
            else:
                output.append(
                    format_playbook_line(
                        merged["id"], merged["helpful"], merged["harmful"], merged["content"]
                    )
                )
        return "\n".join(output)


__all__ = ["BulletpointAnalyzer"]
