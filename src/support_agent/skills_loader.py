"""Skill modules: Markdown + YAML frontmatter, loaded on demand (progressive disclosure)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from support_agent.config import get_settings


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    when_to_use: str
    body: str


@lru_cache
def _all() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for path in sorted(get_settings().skills_dir.glob("*/SKILL.md")):
        text = path.read_text()
        _, fm, body = text.split("---", 2)
        meta = yaml.safe_load(fm)
        skills[meta["name"]] = Skill(
            meta["name"], meta["description"], meta.get("when_to_use", ""), body.strip()
        )
    return skills


def skill_names() -> list[str]:
    return list(_all())


def catalogue() -> str:
    """Cheap catalogue (name + one line) that always lives in the prompt."""
    return "\n".join(f"- {s.name}: {s.description}" for s in _all().values())


def load_skill_body(name: str) -> str:
    s = _all().get(name)
    if s is None:
        return f"Unknown skill '{name}'. Available: {', '.join(skill_names())}"
    return f"## Skill: {s.name}\n{s.body}"


def load_skills(names: list[str]) -> str:
    valid = [n for n in dict.fromkeys(names) if n in _all()]
    return "\n\n".join(load_skill_body(n) for n in valid)
