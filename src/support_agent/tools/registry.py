"""The agent's tools. The model decides which to call; thread context comes from RunnableConfig."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from support_agent.retrieval.hybrid import hybrid_search
from support_agent.skills_loader import load_skill_body
from support_agent.storage import cases
from support_agent.storage.db import db


def _case_for(config: RunnableConfig) -> dict[str, Any] | None:
    thread_id = (config.get("configurable") or {}).get("thread_id")
    return cases.get_case_by_thread(thread_id) if thread_id else None


def _log(config: RunnableConfig, kind: str, payload: dict[str, Any]) -> None:
    case = _case_for(config)
    if case:
        cases.log_event(case["case_id"], kind, payload)


@tool(response_format="content_and_artifact")
def search_similar_tickets(
    query: str,
    config: RunnableConfig,
    queue: str | None = None,
    type: str | None = None,
    priority: str | None = None,
    k: int = 6,
) -> tuple[str, list[dict]]:
    """Search historical (and newly resolved) support tickets similar to `query`.
    Use specific problem wording. Optional filters: queue, type (Incident/Request/Problem/Change),
    priority (low/medium/high). Returns ticket IDs with summaries and the past answer."""
    items = hybrid_search(query, queue=queue, type_=type, priority=priority, k=max(1, min(k, 10)))
    _log(
        config,
        "tool:search_similar_tickets",
        {"query": query, "queue": queue, "type": type, "ids": [i.ticket_id for i in items]},
    )
    if not items:
        return "No similar tickets found.", []
    text = "\n".join(
        f"[{i.ticket_id}|{i.source}|{i.queue}|{i.type}|{i.priority}|sim={i.similarity}] "
        f"PROBLEM: {i.summary[:260]} || ANSWER: {i.answer_snippet[:300]}"
        for i in items
    )
    return text, [i.model_dump() for i in items]


@tool
def get_ticket_by_id(ticket_id: str, config: RunnableConfig) -> str:
    """Fetch the full record (subject, body, answer, classification, tags) of one ticket by ID,
    e.g. 'H-004211' (historical) or 'C-20260101-AB12CD' (live case)."""
    _log(config, "tool:get_ticket_by_id", {"ticket_id": ticket_id})
    with db() as c:
        r = c.execute(
            "SELECT * FROM tickets WHERE ticket_id=? AND holdout=0", (ticket_id.strip(),)
        ).fetchone()
        if r is None:
            r = c.execute(
                "SELECT case_id AS ticket_id, subject, body, resolution AS answer, queue, type, "
                "priority, tags_json, status FROM cases WHERE case_id=?",
                (ticket_id.strip(),),
            ).fetchone()
    if r is None:
        return f"Ticket {ticket_id} not found."
    d = dict(r)
    return json.dumps(
        {
            k: d.get(k)
            for k in (
                "ticket_id",
                "subject",
                "body",
                "answer",
                "queue",
                "type",
                "priority",
                "tags_json",
                "status",
            )
        },
        default=str,
    )[:3000]


@tool
def get_customer_history(config: RunnableConfig, limit: int = 5) -> str:
    """Get this customer's previous and open cases (subject, status, classification, resolution)
    and remembered facts. Use it to detect repeat issues or known environment details."""
    case = _case_for(config)
    if not case:
        return "No customer context available."
    from support_agent.memory.long_term import load_memories

    hist = cases.customer_cases(
        case["customer_id"], limit=max(1, min(limit, 10)), exclude=case["case_id"]
    )
    mem = load_memories(case["customer_id"], case["subject"] + " " + case["body"])
    _log(config, "tool:get_customer_history", {"n_cases": len(hist), "n_memory": len(mem)})
    if not hist and not mem:
        return "No prior history for this customer."
    lines = [
        f"- {h['case_id']} [{h['status']}] {h['subject']} | {h.get('queue')}/{h.get('type')} "
        f"| resolution: {(h.get('resolution') or 'n/a')[:160]}"
        for h in hist
    ]
    lines += [f"- MEMORY({m['kind']}): {m['text']}" for m in mem]
    return "\n".join(lines)


@tool
def get_queue_tag_statistics(
    config: RunnableConfig,
    queue: str | None = None,
    type: str | None = None,
    tag: str | None = None,
) -> str:
    """Aggregate statistics over historical tickets for a queue / type / tag: counts, priority
    mix, top tags and common type. Use it to sanity-check classification or typical handling."""
    sql, params = "SELECT queue, type, priority, tags_json FROM tickets WHERE holdout=0", []
    for col, val in (("queue", queue), ("type", type)):
        if val:
            sql += f" AND {col}=?"
            params.append(val)
    if tag:
        sql += " AND tags_json LIKE ?"
        params.append(f'%"{tag}"%')
    with db() as c:
        rows = c.execute(sql, params).fetchall()
    _log(
        config,
        "tool:get_queue_tag_statistics",
        {"queue": queue, "type": type, "tag": tag, "n": len(rows)},
    )
    if not rows:
        return "No tickets match those filters."
    tags: Counter[str] = Counter()
    for r in rows:
        tags.update(json.loads(r["tags_json"] or "[]"))
    out = {
        "count": len(rows),
        "priority_mix": dict(Counter(r["priority"] for r in rows).most_common()),
        "type_mix": dict(Counter(r["type"] for r in rows).most_common()),
        "queue_mix": dict(Counter(r["queue"] for r in rows).most_common(5)),
        "top_tags": dict(tags.most_common(8)),
    }
    return json.dumps(out)


@tool
def update_classification(
    queue: str,
    type: str,
    priority: str,
    reason: str,
    config: RunnableConfig,
    tags: list[str] | None = None,
) -> str:
    """Revise the open case's classification when evidence shows the first triage was wrong.
    type must be Incident/Request/Problem/Change; priority low/medium/high/critical."""
    case = _case_for(config)
    if not case:
        return "No open case for this thread."
    pr = priority.lower()
    if pr not in {"low", "medium", "high", "critical"}:
        return "Invalid priority."
    if type not in {"Incident", "Request", "Problem", "Change"}:
        return "Invalid type."
    cases.update_classification(case["case_id"], queue, type, pr, tags or [])
    cases.log_event(
        case["case_id"],
        "classification_updated",
        {"queue": queue, "type": type, "priority": pr, "tags": tags, "reason": reason},
    )
    return f"Classification updated to {queue}/{type}/{pr}."


@tool
def escalate_ticket(
    reason: str, target_queue: str, config: RunnableConfig, severity: str = "normal"
) -> str:
    """Request human escalation for the case (no reliable evidence, conflicting evidence, or
    high-risk issue). Records the request; the final decision is made when drafting the result."""
    case = _case_for(config)
    if not case:
        return "No open case for this thread."
    cases.log_event(
        case["case_id"],
        "escalation_requested",
        {"reason": reason, "target_queue": target_queue, "severity": severity},
    )
    return "Escalation requested and recorded."


@tool
def load_skill(name: str) -> str:
    """Load the full instructions of a skill by name (ticket-triage, incident-investigation,
    clarification, escalation, customer-response)."""
    return load_skill_body(name)


INVESTIGATION_TOOLS = [
    search_similar_tickets,
    get_ticket_by_id,
    get_customer_history,
    get_queue_tag_statistics,
    update_classification,
    escalate_ticket,
    load_skill,
]
TOOLS_BY_NAME = {t.name: t for t in INVESTIGATION_TOOLS}
