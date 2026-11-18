"""Long-term (customer-level) memory with an explicit write-worthiness gate.

Policy - store ONLY stable, reusable, customer-specific facts:
  environment (stack / product / setup the customer uses), pattern (recurring problem),
  preference (language, tone, channel), caveat (unresolved constraint), outcome (1-line result
  of an ACCEPTED case).
Never store: raw ticket text, greetings, one-off transient details, secrets/PII, duplicates.
"""

from __future__ import annotations

import re
from typing import Any

from support_agent.storage.db import db, fts_query

MAX_ITEMS_PER_CUSTOMER = 30
MIN_LEN, MAX_LEN = 12, 200

_PII = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"\+?\d[\d\-\s().]{8,}\d"),  # phone-like
    re.compile(r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9_\-]{8,}\b", re.IGNORECASE),
    re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"https?://\S+"),
]
_FILLER = re.compile(
    r"(?i)^(hi|hello|dear|thanks?|thank you|please|regards|sincerely|i am writing|i hope)\b"
)


def scrub(text: str) -> str:
    for p in _PII:
        text = p.sub("[redacted]", text)
    return text.strip()


def is_write_worthy(text: str, kind: str) -> bool:
    t = text.strip()
    if not (MIN_LEN <= len(t) <= MAX_LEN):
        return False
    if "[redacted]" in t or scrub(t) != t or _FILLER.match(t):
        return False
    if kind not in {"environment", "pattern", "preference", "caveat", "outcome"}:
        return False
    return len(t.split()) >= 3


def _norm(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", t.lower()))


def _near_duplicate(a: str, b: str) -> bool:
    x, y = _norm(a), _norm(b)
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= 0.8


_GENERIC = {
    "customer",
    "uses",
    "using",
    "used",
    "their",
    "with",
    "that",
    "this",
    "has",
    "have",
    "from",
    "user",
    "users",
    "company",
    "team",
}


def is_grounded(text: str, source: str, min_ratio: float = 0.5) -> bool:
    """A memory must be supported by what the customer actually wrote (anti-hallucination)."""
    toks = {t for t in re.findall(r"[a-z0-9]{4,}", text.lower()) if t not in _GENERIC}
    if not toks:
        return False
    src = set(re.findall(r"[a-z0-9]{4,}", source.lower()))
    return len(toks & src) / len(toks) >= min_ratio


def save_memories(
    customer_id: str, items: list[dict[str, str]], case_id: str, source_text: str | None = None
) -> list[dict]:
    """Apply the gate and persist. Returns the items actually stored.

    `source_text` (the customer's own words) enables the grounding check; `outcome` items are
    system-generated and exempt.
    """
    stored: list[dict] = []
    with db() as c:
        existing = [
            r["text"]
            for r in c.execute(
                "SELECT text FROM customer_memory WHERE customer_id=?", (customer_id,)
            )
        ]
        for it in items:
            text, kind = scrub(it["text"]), it["kind"]
            if not is_write_worthy(text, kind):
                continue
            if source_text is not None and kind != "outcome" and not is_grounded(text, source_text):
                continue
            if any(_near_duplicate(text, e) for e in existing):
                continue
            c.execute(
                "INSERT OR IGNORE INTO customer_memory(customer_id, kind, text, source_case_id) "
                "VALUES (?,?,?,?)",
                (customer_id, kind, text, case_id),
            )
            existing.append(text)
            stored.append({"kind": kind, "text": text})
        # cap per customer: drop least recently used
        c.execute(
            "DELETE FROM customer_memory WHERE customer_id=? AND id NOT IN ("
            "SELECT id FROM customer_memory WHERE customer_id=? "
            "ORDER BY last_used_at DESC, id DESC LIMIT ?)",
            (customer_id, customer_id, MAX_ITEMS_PER_CUSTOMER),
        )
    return stored


def load_memories(customer_id: str, query: str = "", limit: int = 6) -> list[dict[str, Any]]:
    """Return the most relevant memories (lexical overlap with the new ticket, then recency)."""
    with db() as c:
        rows = [
            dict(r)
            for r in c.execute(
                "SELECT id, kind, text, source_case_id, created_at FROM customer_memory "
                "WHERE customer_id=? ORDER BY last_used_at DESC, id DESC",
                (customer_id,),
            )
        ]
        q = _norm(query)
        for r in rows:
            r["_score"] = len(q & _norm(r["text"])) + (
                0.5 if r["kind"] in ("environment", "preference") else 0
            )
        rows.sort(key=lambda r: r["_score"], reverse=True)
        picked = rows[:limit]
        for r in picked:
            c.execute(
                "UPDATE customer_memory SET last_used_at=CURRENT_TIMESTAMP WHERE id=?", (r["id"],)
            )
    for r in picked:
        r.pop("_score", None)
    return picked


__all__ = ["fts_query", "is_write_worthy", "load_memories", "save_memories", "scrub"]
