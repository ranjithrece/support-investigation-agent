"""Friendly pre-run checks so a first-time user gets an actionable message, not a stack trace."""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request

from support_agent.config import get_settings


class PreflightError(RuntimeError):
    """Raised with a human-readable fix-it message."""


def _ollama_models(base_url: str) -> set[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as r:  # noqa: S310 (local)
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise PreflightError(
            f"Cannot reach Ollama at {base_url}. Install it from https://ollama.com/download "
            "and start it (`ollama serve`, or open the Ollama app), then retry."
        ) from e
    return {m["name"] for m in data.get("models", [])}


def check_ollama(need_chat: bool = True) -> None:
    s = get_settings()
    have = _ollama_models(s.ollama_base_url)

    def present(model: str) -> bool:
        return model in have or f"{model}:latest" in have

    needed = [s.embed_model] + ([s.chat_model] if need_chat else [])
    missing = [m for m in needed if not present(m)]
    if missing:
        raise PreflightError(
            "Missing Ollama model(s): "
            + ", ".join(missing)
            + ". Run: "
            + " && ".join(f"ollama pull {m}" for m in missing)
        )


def check_data() -> None:
    s = get_settings()
    n = 0
    if s.app_db.exists():
        try:
            conn = sqlite3.connect(s.app_db)
            try:
                n = conn.execute("SELECT count(*) FROM tickets WHERE holdout=0").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error:
            n = 0
    if n == 0:
        raise PreflightError(
            "No ticket knowledge base found. Run once: `uv run agent download` then "
            "`uv run agent ingest` (about 15 minutes), and retry."
        )


def run_all(need_chat: bool = True) -> None:
    """Check data first (cheap), then Ollama."""
    check_data()
    check_ollama(need_chat)
