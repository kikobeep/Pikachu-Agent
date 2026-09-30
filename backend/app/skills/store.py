"""技能文件的扫描、解析和资源读取。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import (
    DEFAULT_PROJECT_SKILLS_DIR, DEFAULT_USER_SKILLS_DIR,
    MAX_SKILL_FILE_BYTES, Skill, SkillMetadata, SkillResources, SkillScope,
)
from .utils import (
    SkillParseError, check_skill_dir, is_relative,
    parse_skill_document, safe_skill_file, validate_skill_name,
)


class SkillStore:
    def __init__(
        self,
        user_dir: str | Path = DEFAULT_USER_SKILLS_DIR,
        project_dir: str | Path = DEFAULT_PROJECT_SKILLS_DIR,
    ) -> None:
        self.user_dir = Path(user_dir).expanduser().resolve()
        self.project_dir = Path(project_dir).expanduser().resolve()
        self._diagnostics: list[str] = []

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self.project_dir.mkdir, parents=True, exist_ok=True
        )

    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    async def list_metadata(self) -> tuple[SkillMetadata, ...]:
        """读取两个目录的元数据，不处理同名优先级。"""
        return await asyncio.to_thread(self._list_metadata)

    def _list_metadata(self) -> tuple[SkillMetadata, ...]:
        self._diagnostics.clear()
        entries = []
        for root, scope in (
            (self.project_dir, SkillScope.PROJECT),
            (self.user_dir, SkillScope.USER),
        ):
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.name.startswith('.') or not child.is_dir():
                    continue
                skill_dir = check_skill_dir(root, child.name)
                if skill_dir is None:
                    self._diagnostics.append(f'{child}: 技能目录无效')
                    continue
                try:
                    parsed, skill_file = self._read_document(skill_dir)
                except (OSError, UnicodeError, ValueError) as exc:
                    self._diagnostics.append(f'{child}: {exc}')
                    continue
                entries.append(SkillMetadata(
                    **parsed.model_dump(exclude={'body'}),
                    scope=scope, location=skill_file,
                ))
        return tuple(entries)

    def _read_document(self, skill_dir: Path):
        skill_file = safe_skill_file(skill_dir)
        if skill_file is None:
            raise SkillParseError('SKILL.md 不存在或路径无效')
        if skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
            raise SkillParseError('SKILL.md 超过文件大小限制')
        text = skill_file.read_text(encoding='utf-8')
        return parse_skill_document(text, expected_name=skill_dir.name), skill_file

    async def list_skill_name(self) -> tuple[SkillMetadata, ...]:
        """同名技能优先使用项目版本，按名称排序。"""
        merged: dict[str, SkillMetadata] = {}
        for metadata in await self.list_metadata():
            if metadata.name not in merged or metadata.scope is SkillScope.PROJECT:
                merged[metadata.name] = metadata
        return tuple(merged[name] for name in sorted(merged))

    async def load_metadata(self) -> tuple[SkillMetadata, ...]:
        return await self.list_skill_name()

    async def load(self, name: str) -> Skill | None:
        """按名称加载完整技能，同名时优先使用项目版本。"""
        name = validate_skill_name(name)
        for metadata in await self.list_skill_name():
            if metadata.name == name:
                return await asyncio.to_thread(self._load, metadata)
        return None

    def _load(self, metadata: SkillMetadata) -> Skill | None:
        root = self.project_dir if metadata.scope is SkillScope.PROJECT else self.user_dir
        skill_dir = check_skill_dir(root, metadata.name)
        if skill_dir is None:
            return None
        try:
            parsed, skill_file = self._read_document(skill_dir)
            return Skill(
                metadata=SkillMetadata(
                    **parsed.model_dump(exclude={'body'}),
                    scope=metadata.scope, location=skill_file,
                ),
                content=parsed.body,
                root=skill_dir,
                resources=self._discover_resources(skill_dir),
            )
        except (OSError, UnicodeError, ValueError) as exc:
            self._diagnostics.append(f'{skill_dir}: {exc}')
            return None

    def _discover_resources(self, skill_dir: Path) -> SkillResources:
        return SkillResources(
            scripts=self._list_resource_dir(skill_dir, 'scripts'),
            references=self._list_resource_dir(skill_dir, 'references'),
            assets=self._list_resource_dir(skill_dir, 'assets'),
        )

    @staticmethod
    def _list_resource_dir(skill_dir: Path, subdir: str) -> tuple[str, ...]:
        directory = skill_dir / subdir
        if directory.is_symlink() or not directory.is_dir():
            return ()
        entries = []
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(skill_dir)
            if not path.is_file() or any(
                parent.is_symlink()
                for parent in (path, *path.parents)
                if parent != skill_dir and is_relative(skill_dir, parent)
            ):
                continue
            if is_relative(skill_dir, path.resolve()):
                entries.append(relative.as_posix())
        return tuple(entries)
