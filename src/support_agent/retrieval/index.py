"""Make a trusted, resolved live case searchable for future retrieval (knowledge growth)."""

from __future__ import annotations

import json

from support_agent.ingest.loader import embed_text
from support_agent.retrieval.embeddings import collection, embed_documents
from support_agent.storage.db import db


def index_live_case(case: dict) -> None:
    """Insert the case into tickets/FTS (source='live') and the Chroma index."""
    tid = case["case_id"]
    tags = case.get("tags") or json.loads(case.get("tags_json") or "[]")
    with db() as c:
        c.execute(
            "INSERT OR REPLACE INTO tickets(ticket_id, source, subject, body, answer, queue, type, "
            "priority, language, tags_json, holdout, indexed) VALUES (?,?,?,?,?,?,?,?,?,?,0,1)",
            (
                tid,
                "live",
                case["subject"],
                case["body"],
                case["resolution"],
                case["queue"],
                case["type"],
                case["priority"],
                "en",
                json.dumps(tags),
            ),
        )
        c.execute("DELETE FROM tickets_fts WHERE ticket_id=?", (tid,))
        c.execute(
            "INSERT INTO tickets_fts(ticket_id, subject, body, tags) VALUES (?,?,?,?)",
            (tid, case["subject"], case["body"], " ".join(tags)),
        )
    text = embed_text(case["subject"], case["body"])
    collection().upsert(
        ids=[tid],
        embeddings=embed_documents([text]),
        documents=[text],
        metadatas=[
            {
                "queue": case["queue"],
                "type": case["type"],
                "priority": case["priority"],
                "tags": "|".join(tags),
                "source": "live",
            }
        ],
    )
