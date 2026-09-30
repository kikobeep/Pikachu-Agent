
from __future__ import annotations

from collections.abc import Sequence

from app.model.config import Message, MessageRole
from app.context.tokens import default_token_estimator
from app.skills.config import Skill, SkillMetadata
from app.skills.prompt import _METADATA_HEADER

_DEFAULT_CATALOG_MAX_TOKENS = 2048

class SkillContextProvider:
    def __init__(
        self,
        *,
        max_tokens: int,
        max_active: int,
        max_meta_tokens: int = _DEFAULT_CATALOG_MAX_TOKENS
    ):
        self.max_tokens = max_tokens
        self.max_active = max_active
        self.max_meta_tokens = max_meta_tokens
        self._estimator = default_token_estimator()
    
    def render_meta(self, metadata: Sequence[SkillMetadata]) -> str:
        if not metadata:
            return _METADATA_HEADER + "(No skills available.)\n"
        lines = [_METADATA_HEADER.rstrip()]
        shown = 0
        for item in metadata:
            candidate = lines + [f"[{item.name}] {item.description}"]
            if (
                self._estimator.estimate_text("\n".join(candidate))
                > self.max_meta_tokens
            ):
                break
            lines = candidate
            shown += 1
        hidden = len(metadata) - shown
        if hidden > 0:
            lines.append(f"... {hidden} additional skills are not shown.")
        return "\n".join(lines).rstrip() + "\n"

    def get_meta_message(self, metadata: Sequence[SkillMetadata]) -> Message | None:
        text = self.render_meta(metadata)
        return Message(
            role=MessageRole.SYSTEM,
            name="SKILL_METADATA",
            content=text,
        )
    
    def meta_tokens(
        self,
        metadata: Sequence[SkillMetadata],
    ) -> int:
        return self._estimator.estimate_text(self.render_meta(metadata))


    def active_messages(
        self,
        skills: Sequence[Skill],
    ) -> tuple[Message, ...]:
        """按激活顺序渲染 Active Skill 指令块（去重后）。"""

        seen: set[str] = set()
        messages: list[Message] = []
        for skill in skills:
            if skill.metadata.name in seen:
                continue
            seen.add(skill.metadata.name)
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    name="active skill",
                    content=skill.render_instructions(),
                )
            )
        return tuple(messages)

    def active_tokens(self, skills: Sequence[Skill]) -> int:
        total = 0
        skill_messages = self.active_messages(skills)
        for message in skill_messages:
            total += self._estimator.estimate_text(message.content or "")
        return total

    def check_budget(
        self,
        current: Sequence[Skill],
        candidate: Skill,
    ) -> bool:
        """判断激活 candidate 是否超出总预算（True = 超出/应拒绝）。"""

        if len(current) >= self.max_active:
            return True
        if any(
            skill.metadata.name == candidate.metadata.name for skill in current
        ):
            return False
        current_skills = [*current, candidate]
        return self.active_tokens(current_skills) > self.max_tokens
    
