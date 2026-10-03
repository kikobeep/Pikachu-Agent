"""String Playbook parsing and deterministic ACE operations.

The original ACE keeps the Playbook as a sectioned string and parses bullet
lines only when it needs to update counters or apply Curator deltas. Sidekick
uses the same representation so the full Playbook can be passed to the LLM.
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


def parse_playbook_line(line: str) -> dict[str, Any] | None:
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())
    if not match:
        return None
    return {
        "id": match.group(1),
        "helpful": int(match.group(2)),
        "harmful": int(match.group(3)),
        "content": match.group(4).strip(),
    }


def iter_playbook_bullets(playbook: str) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    section = "OTHERS"
    for line in playbook.strip().splitlines():
        if line.strip().startswith("##"):
            section = line.strip()[2:].strip()
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            parsed["section"] = section
            bullets.append(parsed)
    return bullets


def extract_playbook_bullets(playbook: str, bullet_ids: list[str] | tuple[str, ...]) -> str:
    wanted = set(bullet_ids)
    return "\n".join(
        line for line in playbook.splitlines()
        if (parsed := parse_playbook_line(line)) is not None and parsed["id"] in wanted
    )


def update_bullet_counts(playbook: str, tags: list[dict[str, Any]]) -> str:
    tag_map = {
        str(item.get("id") or item.get("bullet")): str(item.get("tag") or item.get("label", "neutral"))
        for item in tags
        if isinstance(item, dict) and (item.get("id") or item.get("bullet"))
    }
    if not tag_map:
        return playbook

    updated: list[str] = []
    for line in playbook.splitlines():
        parsed = parse_playbook_line(line)
        if parsed is None or parsed["id"] not in tag_map:
            updated.append(line)
            continue
        tag = tag_map[parsed["id"]]
        if tag == "helpful":
            parsed["helpful"] += 1
        elif tag == "harmful":
            parsed["harmful"] += 1
        updated.append(format_playbook_line(parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]))
    return "\n".join(updated)


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def get_next_global_id(playbook: str) -> int:
    numbers = []
    for bullet in iter_playbook_bullets(playbook):
        match = re.search(r"-(\d+)$", bullet["id"])
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def section_slug(section: str) -> str:
    clean = section.lower().strip().replace(" ", "_").replace("&", "and")
    known = {
        "strategies_and_insights": "str",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
    }
    if clean in known:
        return known[clean]
    return "".join(word[0] for word in clean.split("_") if word)[:5] or "misc"


def apply_curator_operations(playbook: str, operations: list[dict[str, Any]], next_id: int) -> tuple[str, int]:
    lines = playbook.splitlines()
    by_id = {bullet["id"]: bullet for bullet in iter_playbook_bullets(playbook)}
    removed: set[str] = set()
    replacements: dict[str, str] = {}
    additions: dict[str, list[str]] = {}

    for operation in operations:
        action = str(operation.get("action", "")).upper()
        item_id = str(operation.get("id", ""))
        if action == "DROP":
            if item_id in by_id:
                removed.add(item_id)
            continue
        if action == "ADD":
            section = str(operation.get("section", "OTHERS"))
            section_key = section.lower().replace(" ", "_").replace("&", "and")
            existing_sections = {
                line.strip()[2:].strip().lower().replace(" ", "_").replace("&", "and")
                for line in lines if line.strip().startswith("##")
            }
            if section_key not in existing_sections:
                section_key = "others"
                section = "OTHERS"
            bullet_id = item_id if item_id and item_id not in by_id else f"{section_slug(section)}-{next_id:05d}"
            next_id += 1
            additions.setdefault(section_key, []).append(format_playbook_line(bullet_id, 0, 0, str(operation["content"]).strip()))
            by_id[bullet_id] = {"id": bullet_id}
            continue
        if action in {"UPDATE", "MERGE"} and item_id in by_id:
            target = by_id[item_id]
            section = str(operation.get("section") or target.get("section", "OTHERS"))
            content = str(operation.get("content") or target.get("content", "")).strip()
            replacements[item_id] = format_playbook_line(item_id, target.get("helpful", 0), target.get("harmful", 0), content)
            if action == "MERGE":
                for merge_id in operation.get("merge_ids", []):
                    merge_id = str(merge_id)
                    if merge_id == item_id or merge_id not in by_id:
                        continue
                    source = by_id[merge_id]
                    replacements[item_id] = format_playbook_line(
                        item_id,
                        int(target.get("helpful", 0)) + int(source.get("helpful", 0)),
                        int(target.get("harmful", 0)) + int(source.get("harmful", 0)),
                        content,
                    )
                    removed.add(merge_id)
            _ = section

    result: list[str] = []
    current_section = "others"
    for line in lines:
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip().lower().replace(" ", "_").replace("&", "and")
            result.append(line)
            result.extend(additions.pop(current_section, []))
            continue
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in removed:
            continue
        result.append(replacements.get(parsed["id"], line) if parsed else line)
    for extra in additions.values():
        result.extend(extra)
    return "\n".join(result), next_id
