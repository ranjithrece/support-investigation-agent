"""Hybrid retrieval: Chroma (semantic, metadata-filtered) + SQLite FTS5 (BM25) fused with RRF."""

from __future__ import annotations

import logging
from typing import Any

from support_agent.config import get_settings
from support_agent.retrieval.embeddings import collection, embed_query
from support_agent.schemas import EvidenceItem
from support_agent.storage.db import db, fts_query

log = logging.getLogger(__name__)
RRF_K = 60
POOL = 30


def _where(queue: str | None, type_: str | None, priority: str | None) -> dict[str, Any] | None:
    conds = [{k: v} for k, v in (("queue", queue), ("type", type_), ("priority", priority)) if v]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _vector(
    query: str, where: dict[str, Any] | None, include_live: bool
) -> list[tuple[str, float]]:
    col = collection()
    if col.count() == 0:
        return []
    kwargs: dict[str, Any] = {
        "query_embeddings": [embed_query(query)],
        "n_results": min(POOL, col.count()),
    }
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    out = []
    for tid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
        if not include_live and meta.get("source") == "live":
            continue
        out.append((tid, 1.0 - float(dist)))
    return out


def _bm25(
    query: str, queue: str | None, type_: str | None, priority: str | None, include_live: bool
) -> list[str]:
    q = fts_query(query)
    if not q:
        return []
    sql = (
        "SELECT f.ticket_id FROM tickets_fts f JOIN tickets t ON t.ticket_id=f.ticket_id "
        "WHERE tickets_fts MATCH ? AND t.holdout=0"
    )
    params: list[Any] = [q]
    for col_name, val in (("queue", queue), ("type", type_), ("priority", priority)):
        if val:
            sql += f" AND t.{col_name}=?"
            params.append(val)
    if not include_live:
        sql += " AND t.source='hist'"
    sql += " ORDER BY rank LIMIT ?"
    params.append(POOL)
    with db() as c:
        return [r[0] for r in c.execute(sql, params)]


def _hydrate(ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    with db() as c:
        rows = c.execute(
            f"SELECT * FROM tickets WHERE ticket_id IN ({','.join('?' * len(ids))})", ids
        ).fetchall()
    return {r["ticket_id"]: dict(r) for r in rows}


def _item(rec: dict[str, Any], sim: float) -> EvidenceItem:
    body = (rec.get("body") or "").replace("\n", " ")
    return EvidenceItem(
        ticket_id=rec["ticket_id"],
        source=rec["source"],
        queue=rec.get("queue"),
        type=rec.get("type"),
        priority=rec.get("priority"),
        similarity=round(sim, 3),
        summary=f"{rec.get('subject') or ''}: {body[:280]}".strip(": "),
        answer_snippet=(rec.get("answer") or "").replace("\n", " ")[:500],
    )


def hybrid_search(
    query: str,
    queue: str | None = None,
    type_: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    k: int | None = None,
    include_live: bool = True,
    min_hits: int = 3,
) -> list[EvidenceItem]:
    """Return top-k evidence. Filters are relaxed (priority, then type, then queue) when too few hits."""
    k = k or get_settings().retrieval_k
    relax_order = [
        (queue, type_, priority),
        (queue, type_, None),
        (queue, None, None),
        (None, None, None),
    ]
    seen: set[tuple] = set()
    fused: dict[str, float] = {}
    sims: dict[str, float] = {}
    for q_, t_, p_ in relax_order:
        if (q_, t_, p_) in seen:
            continue
        seen.add((q_, t_, p_))
        vec = _vector(query, _where(q_, t_, p_), include_live)
        bm = _bm25(query, q_, t_, p_, include_live)
        fused, sims = {}, {}
        for rank, (tid, sim) in enumerate(vec):
            fused[tid] = fused.get(tid, 0) + 1 / (RRF_K + rank + 1)
            sims[tid] = sim
        for rank, tid in enumerate(bm):
            fused[tid] = fused.get(tid, 0) + 1 / (RRF_K + rank + 1)
        if len(fused) >= min_hits:
            if (q_, t_, p_) != (queue, type_, priority):
                log.info("retrieval filters relaxed to %s", (q_, t_, p_))
            break

    recs = _hydrate(list(fused))
    want = {t.lower() for t in (tags or [])}
    if want:  # small boost for tag overlap
        import json

        for tid, rec in recs.items():
            have = {t.lower() for t in json.loads(rec.get("tags_json") or "[]")}
            if have & want:
                fused[tid] += 0.004 * len(have & want)
    ranked = sorted(fused, key=lambda t: fused[t], reverse=True)[:k]
    out = []
    for tid in ranked:
        if tid in recs and not recs[tid].get("holdout"):
            sim = sims.get(tid)
            if sim is None:  # BM25-only hit: derive a soft score from the fusion value
                sim = min(0.5, fused[tid] * 20)
            out.append(_item(recs[tid], sim))
    return out
