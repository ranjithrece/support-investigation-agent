"""Ingest the historical dataset: SQLite (+FTS5) and Chroma. Idempotent and resumable."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress

from support_agent.config import get_settings
from support_agent.ingest.loader import assign_holdout, embed_text, load_english
from support_agent.retrieval.embeddings import collection, embed_documents
from support_agent.storage.db import db, init_db

console = Console()
BATCH = 64


def _default_csvs() -> list[Path]:
    raw = get_settings().data_dir / "raw"
    return sorted(raw.glob("*.csv"))


def ingest(
    csv_paths: list[Path] | None = None,
    limit: int | None = None,
    holdout: int = 200,
    rebuild: bool = False,
    skip_vectors: bool = False,
) -> dict[str, int]:
    s = get_settings()
    csv_paths = csv_paths or _default_csvs()
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files in {s.data_dir / 'raw'} - run `agent download` first"
        )

    if rebuild:
        for f in (s.app_db, Path(str(s.app_db) + "-wal"), Path(str(s.app_db) + "-shm")):
            f.unlink(missing_ok=True)
        import shutil

        shutil.rmtree(s.chroma_dir, ignore_errors=True)
    init_db()

    df = load_english(csv_paths)
    df = assign_holdout(df, holdout)
    if limit:
        # keep holdout rows inside the limited sample so eval still works
        hist = df[df["holdout"] == 0].head(limit)
        held = df[df["holdout"] == 1].head(max(10, limit // 10))
        df = pd.concat([hist, held])
    console.print(f"[bold]{len(df)}[/] English tickets ({int(df.holdout.sum())} hold-out)")

    with db() as c:
        existing = {r[0] for r in c.execute("SELECT ticket_id FROM tickets WHERE source='hist'")}
        new = df[~df["ticket_id"].isin(existing)]
        for r in new.itertuples():
            c.execute(
                "INSERT OR IGNORE INTO tickets(ticket_id, source, subject, body, answer, queue, "
                "type, priority, language, tags_json, holdout) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r.ticket_id,
                    "hist",
                    r.subject,
                    r.body,
                    r.answer,
                    r.queue,
                    r.type,
                    r.priority,
                    r.language,
                    json.dumps(r.tags),
                    int(r.holdout),
                ),
            )
            if not r.holdout:
                c.execute(
                    "INSERT INTO tickets_fts(ticket_id, subject, body, tags) VALUES (?,?,?,?)",
                    (r.ticket_id, r.subject, r.body, " ".join(r.tags)),
                )
    stats = {"tickets": len(df), "new_rows": len(new), "embedded": 0}
    if skip_vectors:
        return stats

    col = collection()
    done = set()
    got = col.get(include=[])
    done.update(got["ids"])
    todo = df[(df["holdout"] == 0) & (~df["ticket_id"].isin(done))]
    console.print(
        f"Embedding {len(todo)} tickets with [cyan]{s.embed_model}[/] ({len(done)} already)"
    )
    t0 = time.time()
    with Progress(console=console) as prog:
        task = prog.add_task("embedding", total=len(todo))
        rows = list(todo.itertuples())
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            vecs = embed_documents([embed_text(r.subject, r.body) for r in chunk])
            col.upsert(
                ids=[r.ticket_id for r in chunk],
                embeddings=vecs,
                documents=[embed_text(r.subject, r.body) for r in chunk],
                metadatas=[
                    {
                        "queue": r.queue,
                        "type": r.type,
                        "priority": r.priority,
                        "tags": "|".join(r.tags),
                        "source": "hist",
                    }
                    for r in chunk
                ],
            )
            prog.advance(task, len(chunk))
    with db() as c:
        c.execute("UPDATE tickets SET indexed=1 WHERE source='hist' AND holdout=0")
    stats["embedded"] = len(todo)
    console.print(f"Done in {time.time() - t0:.0f}s")
    return stats
