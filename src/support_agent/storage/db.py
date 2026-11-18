"""SQLite access helpers (stdlib only, parameterised queries)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from support_agent.config import get_settings

_SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or get_settings().app_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(_SCHEMA.read_text())


@contextmanager
def db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("tags_json", "evidence_json"):
        if k in d and isinstance(d[k], str):
            d[k.removesuffix("_json")] = json.loads(d[k] or "[]")
    return d


def fts_query(text: str, max_terms: int = 24) -> str:
    """Turn free text into a safe FTS5 OR-query (alphanumeric tokens only)."""
    import re

    toks = [t for t in re.findall(r"[A-Za-z0-9]{3,}", text.lower())]
    seen: list[str] = []
    for t in toks:
        if t not in seen:
            seen.append(t)
    return " OR ".join(f'"{t}"' for t in seen[:max_terms])
