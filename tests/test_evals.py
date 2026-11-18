"""Evaluator tests: code-based evaluators are exercised against stubbed runs and a real graph run."""

from __future__ import annotations

from types import SimpleNamespace

from support_agent.evals import runner as R
from support_agent.storage import cases

GOOD = {
    "case_id": "C-1",
    "classification": {
        "queue": "Billing and Payments",
        "type": "Request",
        "priority": "medium",
        "tags": [],
    },
    "evidence": [
        {
            "ticket_id": "H-000004",
            "source": "hist",
            "queue": "Billing and Payments",
            "type": "Request",
            "priority": "medium",
            "similarity": 0.8,
            "summary": "s",
            "answer_snippet": "a",
        }
    ],
    "analysis": "H-000004 shows a refund of the duplicate charge.",
    "resolution": "We will refund it.",
    "escalation": {"required": False, "reason": None, "target_queue": None},
    "confidence": 0.8,
    "verification": {"grounded": True, "grounding_score": 0.9, "unsupported_claims": []},
}


def run_of(**o):
    return SimpleNamespace(outputs=o)


def ex(**o):
    return SimpleNamespace(outputs=o, inputs={"subject": "s", "body": "b"})


def test_retrieval_and_groundedness_scores():
    run = run_of(result=GOOD)
    assert R.retrieval_relevance(run, ex(reference_queue="Billing and Payments"))["score"] > 0.9
    assert R.retrieval_relevance(run, ex(reference_queue="Technical Support"))["score"] < 0.6
    assert R.groundedness(run, ex())["score"] > 0.8
    assert R.retrieval_relevance(run_of(result=None), ex())["score"] == 0.0


def test_tool_usage_flags_redundant_and_unjustified_escalation():
    ev = [
        {"kind": "tool_call", "payload": {"name": "search_similar_tickets", "args": {"query": "x"}}}
    ] * 2 + [{"kind": "tool_call", "payload": {"name": "escalate_ticket", "args": {}}}]
    s = R.tool_usage(run_of(result=GOOD, events=ev), ex())["score"]
    assert s == 0.5  # searched, but redundant + escalation not justified
    clean = [
        {"kind": "tool_call", "payload": {"name": "search_similar_tickets", "args": {"query": "x"}}}
    ]
    assert R.tool_usage(run_of(result=GOOD, events=clean), ex())["score"] == 1.0


def test_behaviour_checks_clarification_and_escalation_expectations():
    o = dict(result=GOOD, completed=True, clarifications_asked=0, events=[])
    assert (
        R.behaviour(run_of(**o), ex(expect_clarification=False, expect_escalation=False))["score"]
        == 1.0
    )
    assert R.behaviour(run_of(**o), ex(expect_clarification=True))["score"] < 1.0
    assert R.behaviour(run_of(**o), ex(expect_escalation=True))["score"] < 1.0


def test_build_examples_uses_only_holdout(tmp_path):
    from support_agent.storage.db import db

    with db() as c:
        c.execute("UPDATE tickets SET holdout=1 WHERE ticket_id IN ('H-000004','H-000006')")
    exs = R.build_examples(limit=5)
    sources = {e["outputs"].get("source_ticket") for e in exs if e["outputs"].get("source_ticket")}
    assert sources <= {"H-000004", "H-000006"} and len(exs) >= len(R.BEHAVIOUR_CASES)


def test_target_and_evaluators_end_to_end(install_llm):
    install_llm()
    out = R.make_target()(
        {"subject": "VPN keeps dropping", "body": "VPN disconnects every few minutes."}
    )
    assert out["completed"] and out["result"] and out["events"]
    run = run_of(**out)
    for fn in (R.retrieval_relevance, R.groundedness, R.tool_usage, R.behaviour):
        res = fn(run, ex(reference_queue="Technical Support", expect_clarification=False))
        assert 0.0 <= res["score"] <= 1.0
    assert cases.get_case(out["case_id"])
