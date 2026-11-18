"""LangSmith evaluators + datasets, run locally (upload_results=False => no account / key needed).

Evaluators (all follow the LangSmith (run, example) -> {key, score, comment} contract):
  retrieval_relevance  code   - hit@k of the reference queue/type among evidence + mean similarity
  groundedness         code   - verifier grounding score + citations point at retrieved evidence
  response_quality     judge  - local Ollama LLM-as-judge vs the reference answer (1-5 -> 0..1)
  tool_usage           code   - retrieval tool used, no redundant calls, escalate only if required
  behaviour            code   - schema valid, clarification asked iff dataset says ambiguous,
                                loops bounded, escalation expectation met
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from support_agent.config import get_settings
from support_agent.llm import structured
from support_agent.schemas import CaseResult
from support_agent.storage import cases
from support_agent.storage.db import db

# --------------------------------------------------------------------------- dataset
BEHAVIOUR_CASES: list[dict[str, Any]] = [
    {
        "inputs": {
            "subject": "It does not work",
            "body": "Hi, it is not working again. Please fix.",
        },
        "outputs": {
            "expect_clarification": True,
            "expect_escalation": None,
            "reference_queue": None,
        },
    },
    {
        "inputs": {"subject": "Help", "body": "I cannot access it since this morning."},
        "outputs": {
            "expect_clarification": True,
            "expect_escalation": None,
            "reference_queue": None,
        },
    },
    {
        "inputs": {
            "subject": "Quantum entanglement billing for my spaceship fleet",
            "body": "Please recalibrate the flux capacitor invoice for Mars colony 7.",
        },
        "outputs": {
            "expect_clarification": False,
            "expect_escalation": True,
            "reference_queue": None,
        },
    },
    {
        "inputs": {
            "subject": "Charged twice for subscription",
            "body": "My credit card was charged twice for this month's subscription invoice. Please refund the duplicate payment.",
        },
        "outputs": {
            "expect_clarification": False,
            "expect_escalation": False,
            "reference_queue": "Billing and Payments",
        },
    },
]


def build_examples(limit: int = 30, seed: int = 7) -> list[dict[str, Any]]:
    """Hold-out tickets (never indexed) + hand-written behaviour cases."""
    with db() as c:
        rows = c.execute(
            "SELECT ticket_id, subject, body, answer, queue, type FROM tickets "
            "WHERE holdout=1 AND length(body) > 80 ORDER BY (abs(random() + ?) % 1000003) LIMIT ?",
            (seed, limit),
        ).fetchall()
    ex = [
        {
            "inputs": {"subject": r["subject"], "body": r["body"]},
            "outputs": {
                "reference_answer": r["answer"],
                "reference_queue": r["queue"],
                "reference_type": r["type"],
                "expect_clarification": None,
                "expect_escalation": None,
                "source_ticket": r["ticket_id"],
            },
        }
        for r in rows
    ]
    return ex + BEHAVIOUR_CASES


# --------------------------------------------------------------------------- target
def _run_agent(inputs: dict[str, Any], auto_clarify: str) -> dict[str, Any]:
    from support_agent import service

    customer = f"eval-{uuid.uuid4().hex[:6]}"
    env = service.start_ticket(customer, inputs["subject"], inputs["body"], auto_accept=True)
    asked = 0
    while env["status"] == "needs_input" and asked < 3:
        asked += 1
        env = service.resume(env["thread_id"], auto_clarify)
    events = cases.case_events(env["case_id"]) if env.get("case_id") else []
    return {
        "result": env.get("result"),
        "case_id": env.get("case_id"),
        "clarifications_asked": asked,
        "events": events,
        "completed": env["status"] == "completed",
    }


def make_target(auto_clarify: str = "Please proceed with the information provided in the ticket."):
    """Run the agent end-to-end (auto-accept). Answers a clarification once so the run completes."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        from langsmith import tracing_context

        # The agent's own LangGraph/LangChain runs must not be posted to hosted LangSmith.
        with tracing_context(enabled=False):
            return _run_agent(inputs, auto_clarify)

    return target


# --------------------------------------------------------------------------- evaluators
def _out(run) -> dict[str, Any]:
    return run.outputs or {}


def retrieval_relevance(run, example) -> dict[str, Any]:
    r = _out(run).get("result")
    ref = example.outputs or {}
    if not r or not r["evidence"]:
        return {"key": "retrieval_relevance", "score": 0.0, "comment": "no evidence"}
    ev = r["evidence"]
    q, t = ref.get("reference_queue"), ref.get("reference_type")
    parts = [min(1.0, sum(e["similarity"] for e in ev) / len(ev) / 0.7)]
    if q:
        parts.append(sum(1 for e in ev if e["queue"] == q) / len(ev))
    if t:
        parts.append(sum(1 for e in ev if e["type"] == t) / len(ev))
    return {
        "key": "retrieval_relevance",
        "score": round(sum(parts) / len(parts), 3),
        "comment": f"queue_hit_rate/type_hit_rate over {len(ev)} evidence items",
    }


def groundedness(run, example) -> dict[str, Any]:
    r = _out(run).get("result")
    if not r:
        return {"key": "groundedness", "score": 0.0, "comment": "no result"}
    v = r.get("verification") or {}
    score = float(v.get("grounding_score", 0.0))
    ids = {e["ticket_id"] for e in r["evidence"]}
    cited = set(re.findall(r"\b(?:H-\d{6}|C-\d{8}-[A-Z0-9]{6})\b", r["analysis"]))
    if cited:
        score *= 0.5 + 0.5 * (len(cited & ids) / len(cited))
    elif not r["escalation"]["required"]:
        score *= 0.7  # analysis must cite evidence IDs
    return {
        "key": "groundedness",
        "score": round(score, 3),
        "comment": f"verifier={v.get('grounding_score')}, cited={sorted(cited)}",
    }


class _Judge(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = wrong/unhelpful ... 5 = correct and complete")
    reason: str = ""


def response_quality(run, example) -> dict[str, Any]:
    r = _out(run).get("result")
    ref = (example.outputs or {}).get("reference_answer")
    if not r:
        return {"key": "response_quality", "score": 0.0, "comment": "no result"}
    if not ref:
        return {
            "key": "response_quality",
            "score": None,
            "comment": "no reference (behaviour case)",
        }
    system = (
        "You grade a support reply against a reference reply for the same ticket. Score 1-5: "
        "5 = same actionable guidance as reference, 3 = partially overlapping, 1 = wrong or "
        "irrelevant. Penalise invented specifics. Be strict."
    )
    user = (
        f"TICKET: {(example.inputs or {}).get('subject')}\n{(example.inputs or {}).get('body', '')[:700]}"
        f"\n\nREFERENCE REPLY:\n{ref[:900]}\n\nCANDIDATE REPLY:\n{r['resolution'][:900]}"
    )
    try:
        j = structured(_Judge, system, user, model=get_settings().judge_model, retries=1)
        return {
            "key": "response_quality",
            "score": round((j.score - 1) / 4, 2),
            "comment": j.reason[:200],
        }
    except RuntimeError:
        return {"key": "response_quality", "score": None, "comment": "judge failed"}


def tool_usage(run, example) -> dict[str, Any]:
    o = _out(run)
    ev = o.get("events") or []
    tools = [e["payload"].get("name") for e in ev if e["kind"] == "tool_call"]
    searched = (
        any(e["kind"] == "tool:search_similar_tickets" for e in ev)
        or "search_similar_tickets" in tools
    )
    sigs = [json.dumps(e["payload"], sort_keys=True) for e in ev if e["kind"] == "tool_call"]
    redundant = len(sigs) - len(set(sigs))
    r = o.get("result")
    esc_called = "escalate_ticket" in tools
    esc_ok = (not esc_called) or bool(r and r["escalation"]["required"])
    score = (
        (0.5 if searched else 0.0) + (0.25 if redundant == 0 else 0.0) + (0.25 if esc_ok else 0.0)
    )
    return {
        "key": "tool_usage",
        "score": score,
        "comment": f"tools={tools}, redundant={redundant}, searched={searched}",
    }


def behaviour(run, example) -> dict[str, Any]:
    o, ref = _out(run), example.outputs or {}
    checks: dict[str, bool] = {}
    r = o.get("result")
    try:
        CaseResult.model_validate(r)
        checks["schema_valid"] = True
    except Exception:  # noqa: BLE001
        checks["schema_valid"] = False
    checks["completed"] = bool(o.get("completed"))
    checks["loops_bounded"] = sum(1 for e in o.get("events", []) if e["kind"] == "assess") <= 6
    if ref.get("expect_clarification") is not None:
        checks["clarification_expected"] = (o.get("clarifications_asked", 0) > 0) == ref[
            "expect_clarification"
        ]
    if ref.get("expect_escalation") is not None and r:
        checks["escalation_expected"] = r["escalation"]["required"] == ref["expect_escalation"]
    if ref.get("reference_queue") and ref.get("expect_clarification") is False and r:
        checks["queue_correct"] = r["classification"]["queue"] == ref["reference_queue"]
    score = sum(checks.values()) / len(checks)
    return {"key": "behaviour", "score": round(score, 2), "comment": json.dumps(checks)}


EVALUATORS = [retrieval_relevance, groundedness, response_quality, tool_usage, behaviour]


# --------------------------------------------------------------------------- runner
def run_evaluation(limit: int = 10, prefix: str = "support-agent") -> list[dict[str, Any]]:
    import os

    # Local-only evaluation: never contact the hosted LangSmith API (no account / key in a POC).
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langsmith import evaluate, schemas

    examples = [
        schemas.Example(id=uuid.uuid4(), inputs=e["inputs"], outputs=e["outputs"])
        for e in build_examples(limit)
    ]
    results = evaluate(
        make_target(),
        data=examples,
        evaluators=EVALUATORS,
        experiment_prefix=prefix,
        upload_results=False,
        max_concurrency=1,
    )
    rows: list[dict[str, Any]] = []
    for row in results:
        er = row["evaluation_results"]["results"]
        rows.append(
            {
                "subject": row["example"].inputs["subject"][:50],
                "scores": {x.key: x.score for x in er},
                "comments": {x.key: x.comment for x in er},
            }
        )
    return rows
