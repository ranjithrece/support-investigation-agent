"""High-level API used by the CLI, demo and evaluators (start / resume a ticket thread)."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import Command

from support_agent.graph.builder import get_app
from support_agent.storage import cases
from support_agent.storage.db import init_db


def _envelope(app, config: dict, out: dict[str, Any]) -> dict[str, Any]:
    """Normalise a graph run into {status, thread_id, case_id, interrupt?, result?}."""
    thread_id = config["configurable"]["thread_id"]
    snap = app.get_state(config)
    values = snap.values or {}
    env: dict[str, Any] = {"thread_id": thread_id, "case_id": values.get("case_id")}
    pending = [i for t in snap.tasks for i in t.interrupts]
    if pending:
        payload = pending[0].value
        env.update(status="needs_input", interrupt=payload)
    else:
        env.update(
            status="completed",
            result=values.get("result"),
            indexed=values.get("indexed", False),
            accepted=values.get("user_accepted"),
            stored_memories=values.get("stored_memories", []),
        )
    return env


def start_ticket(
    customer_id: str,
    subject: str,
    body: str,
    thread_id: str | None = None,
    auto_accept: bool = False,
    app=None,
) -> dict[str, Any]:
    init_db()
    app = app or get_app()
    thread_id = thread_id or f"T-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    out = app.invoke(
        {"customer_id": customer_id, "subject": subject, "body": body, "auto_accept": auto_accept},
        config,
    )
    return _envelope(app, config, out)


def resume(thread_id: str, answer: Any, app=None) -> dict[str, Any]:
    app = app or get_app()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    out = app.invoke(Command(resume=answer), config)
    return _envelope(app, config, out)


def pending_interrupt(thread_id: str, app=None) -> dict[str, Any] | None:
    app = app or get_app()
    snap = app.get_state({"configurable": {"thread_id": thread_id}})
    pend = [i for t in snap.tasks for i in t.interrupts]
    return pend[0].value if pend else None


def show_case(case_id: str) -> dict[str, Any] | None:
    case = cases.get_case(case_id)
    if case:
        case["events"] = cases.case_events(case_id)
    return case
