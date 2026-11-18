"""Case lifecycle persistence: open case -> final case -> trusted/indexed."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from support_agent.storage.db import db, row_to_dict


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_case_id() -> str:
    return f"C-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def create_open_case(customer_id: str, thread_id: str, subject: str, body: str) -> str:
    case_id = new_case_id()
    with db() as c:
        c.execute("INSERT OR IGNORE INTO customers(customer_id) VALUES (?)", (customer_id,))
        c.execute(
            "INSERT INTO cases(case_id, customer_id, thread_id, status, subject, body) "
            "VALUES (?,?,?,?,?,?)",
            (case_id, customer_id, thread_id, "OPEN", subject, body),
        )
    log_event(case_id, "case_opened", {"customer_id": customer_id})
    return case_id


def get_case(case_id: str) -> dict[str, Any] | None:
    with db() as c:
        return row_to_dict(c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())


def get_case_by_thread(thread_id: str) -> dict[str, Any] | None:
    with db() as c:
        return row_to_dict(
            c.execute(
                "SELECT * FROM cases WHERE thread_id=? ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        )


def set_status(case_id: str, status: str) -> None:
    with db() as c:
        c.execute(
            "UPDATE cases SET status=?, updated_at=? WHERE case_id=?", (status, _now(), case_id)
        )


def update_classification(
    case_id: str, queue: str, type_: str, priority: str, tags: list[str]
) -> None:
    with db() as c:
        c.execute(
            "UPDATE cases SET queue=?, type=?, priority=?, tags_json=?, updated_at=? "
            "WHERE case_id=?",
            (queue, type_, priority, json.dumps(tags), _now(), case_id),
        )


def save_proposal(case_id: str, result: dict[str, Any]) -> None:
    cls, esc = result["classification"], result["escalation"]
    with db() as c:
        c.execute(
            "UPDATE cases SET status='PROPOSED', queue=?, type=?, priority=?, tags_json=?, "
            "resolution=?, analysis=?, confidence=?, escalation_required=?, escalation_reason=?, "
            "escalation_target=?, evidence_json=?, updated_at=? WHERE case_id=?",
            (
                cls["queue"],
                cls["type"],
                cls["priority"],
                json.dumps(cls["tags"]),
                result["resolution"],
                result["analysis"],
                result["confidence"],
                int(esc["required"]),
                esc.get("reason"),
                esc.get("target_queue"),
                json.dumps(result["evidence"]),
                _now(),
                case_id,
            ),
        )


def finalize(case_id: str, accepted: bool, escalated: bool) -> None:
    status = "ESCALATED" if escalated else ("RESOLVED" if accepted else "OPEN")
    with db() as c:
        c.execute(
            "UPDATE cases SET status=?, accepted=?, updated_at=? WHERE case_id=?",
            (status, int(accepted), _now(), case_id),
        )


def mark_indexed(case_id: str) -> None:
    with db() as c:
        c.execute(
            "UPDATE cases SET status='INDEXED', indexed=1, updated_at=? WHERE case_id=?",
            (_now(), case_id),
        )


def customer_cases(customer_id: str, limit: int = 10, exclude: str | None = None) -> list[dict]:
    with db() as c:
        rows = c.execute(
            "SELECT * FROM cases WHERE customer_id=? AND case_id != ? "
            "ORDER BY created_at DESC LIMIT ?",
            (customer_id, exclude or "", limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]


def log_event(case_id: str, kind: str, payload: dict[str, Any] | None = None) -> None:
    with db() as c:
        c.execute(
            "INSERT INTO case_events(case_id, kind, payload_json) VALUES (?,?,?)",
            (case_id, kind, json.dumps(payload or {}, default=str)),
        )


def case_events(case_id: str) -> list[dict[str, Any]]:
    with db() as c:
        rows = c.execute(
            "SELECT kind, payload_json, ts FROM case_events WHERE case_id=? ORDER BY id", (case_id,)
        ).fetchall()
    return [
        {"kind": r["kind"], "payload": json.loads(r["payload_json"] or "{}"), "ts": r["ts"]}
        for r in rows
    ]
