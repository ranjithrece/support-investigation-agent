"""Integration tests: the full LangGraph with a scripted LLM, real SQLite/Chroma/FTS, real routing."""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from support_agent import service
from support_agent.graph.builder import build_graph
from support_agent.retrieval.hybrid import hybrid_search
from support_agent.storage import cases
from support_agent.storage.db import db

TICKET = dict(
    subject="VPN keeps dropping",
    body="My VPN disconnects every few minutes and I lose the shared drive.",
)


def app_for(tmp_path, name="cp.db"):
    return build_graph(SqliteSaver(sqlite3.connect(tmp_path / name, check_same_thread=False)))


VAGUE = dict(
    subject="It does not work", body="Hi, it is not working again, please fix zebra quokka."
)


def run(app, customer="cust-1", auto=True, ticket=None, **kw):
    t = ticket or TICKET
    return service.start_ticket(customer, t["subject"], t["body"], auto_accept=auto, app=app, **kw)


def event_kinds(case_id):
    return [e["kind"] for e in cases.case_events(case_id)]


def test_normal_resolution_persists_and_indexes(tmp_path, install_llm):
    install_llm()
    env = run(app_for(tmp_path))
    r = env["result"]
    assert env["status"] == "completed"
    # minimum output contract
    for key in ("classification", "evidence", "analysis", "resolution", "escalation", "confidence"):
        assert key in r
    assert r["classification"]["queue"] == "Technical Support"
    assert r["evidence"] and not r["escalation"]["required"] and r["confidence"] >= 0.7
    case = cases.get_case(r["case_id"])
    assert case["status"] == "INDEXED" and case["indexed"] == 1
    kinds = event_kinds(r["case_id"])
    assert kinds[0] == "case_opened" and "tool_call" in kinds and "compare_evidence" in kinds
    assert "indexed" in kinds


def test_new_case_is_stored_open_before_resolution(tmp_path, install_llm):
    install_llm(assess=[{"route": "ask_user", "hypothesis": "h", "question": "Which device?"}])
    env = run(app_for(tmp_path), auto=False, ticket=VAGUE)
    assert env["status"] == "needs_input"
    c = cases.get_case(env["case_id"])
    assert c["status"] == "AWAITING_USER" and c["subject"] == VAGUE["subject"]


def test_clarification_interrupt_resume_across_app_instances(tmp_path, install_llm):
    install_llm(
        assess=[
            {
                "route": "ask_user",
                "hypothesis": "unknown device",
                "question": "Does it happen on all devices?",
            },
            {"route": "resolve", "hypothesis": "client outdated"},
        ]
    )
    env = run(app_for(tmp_path), auto=True, thread_id="T-clar", ticket=VAGUE)
    assert env["status"] == "needs_input" and env["interrupt"]["type"] == "clarification"
    assert "all devices" in env["interrupt"]["question"]
    # a brand-new graph object on the same SQLite checkpoint file == process restart
    env2 = service.resume("T-clar", "Only on my new laptop", app=app_for(tmp_path))
    assert env2["status"] == "completed" and env2["result"]["resolution"]
    kinds = event_kinds(env["case_id"])
    assert "clarification" in kinds and "followup_retrieval" in kinds


def test_ask_user_is_vetoed_when_evidence_is_strong(tmp_path, install_llm):
    install_llm(
        assess=[{"route": "ask_user", "hypothesis": "h", "question": "Which router model?"}]
    )
    env = run(app_for(tmp_path), auto=True)  # clear VPN ticket => high agreement
    assert env["status"] == "completed"  # guardrail overrides the LLM's unnecessary question


def test_repeated_question_is_not_asked_twice(tmp_path, install_llm):
    q = {"route": "ask_user", "hypothesis": "h", "question": "Which device model are you using?"}
    install_llm(assess=[q, q, q])
    app = app_for(tmp_path)
    env = run(app, auto=True, thread_id="T-rep", ticket=VAGUE)
    assert env["interrupt"]["type"] == "clarification"
    env = service.resume("T-rep", "A Dell laptop", app=app)
    assert env["status"] == "completed"  # same question again => guardrail resolves/escalates


def test_retrieve_more_loop_then_resolve(tmp_path, install_llm):
    install_llm(
        assess=[
            {
                "route": "retrieve_more",
                "hypothesis": "outdated client",
                "followup_query": "update VPN client tunnel protocol",
            },
            {"route": "resolve", "hypothesis": "outdated client"},
        ]
    )
    env = run(app_for(tmp_path))
    assert env["status"] == "completed"
    assert event_kinds(env["case_id"]).count("assess") == 2
    assert "followup_retrieval" in event_kinds(env["case_id"])


def test_verify_failure_loops_back_then_passes(tmp_path, install_llm):
    install_llm(
        verify=[
            {"grounding_score": 0.2, "unsupported_claims": ["a", "b", "c"]},
            {"grounding_score": 0.95, "unsupported_claims": []},
        ]
    )
    env = run(app_for(tmp_path))
    kinds = event_kinds(env["case_id"])
    assert kinds.count("verify") == 2 and env["result"]["verification"]["grounded"]


def test_persistent_unsupported_answer_ends_in_escalation(tmp_path, install_llm):
    bad = {"grounding_score": 0.1, "unsupported_claims": ["x", "y", "z"]}
    install_llm(
        verify=[bad],
        draft=[
            {"analysis": "a", "resolution": "unsupported answer", "cited_ticket_ids": ["H-000001"]},
            {"analysis": "a", "resolution": "unsupported answer", "cited_ticket_ids": ["H-000001"]},
            {"analysis": "a", "resolution": "unsupported answer", "cited_ticket_ids": ["H-000001"]},
            {
                "analysis": "no reliable evidence",
                "resolution": "A specialist will follow up.",
                "cited_ticket_ids": [],
                "escalation_required": True,
                "escalation_reason": "unsupported",
                "target_queue": "Technical Support",
            },
        ],
    )
    env = run(app_for(tmp_path))
    assert env["result"]["escalation"]["required"]
    assert cases.get_case(env["case_id"])["status"] == "ESCALATED"
    assert not env["indexed"]  # escalations never enter the trusted index


def test_escalation_route(tmp_path, install_llm):
    install_llm(
        assess=[{"route": "escalate", "hypothesis": "none", "reason": "no evidence"}],
        draft=[
            {
                "analysis": "Evidence conflicts.",
                "resolution": "We have escalated this to a specialist.",
                "cited_ticket_ids": [],
                "escalation_required": True,
                "escalation_reason": "conflicting evidence",
                "target_queue": "Product Support",
            }
        ],
    )
    env = run(app_for(tmp_path))
    r = env["result"]
    assert r["escalation"]["required"] and r["escalation"]["target_queue"] == "Product Support"
    assert r["confidence"] <= 0.4 and not env["indexed"]


def test_user_rejection_loops_back_to_investigation(tmp_path, install_llm):
    llm = install_llm()
    app = app_for(tmp_path)
    env = run(app, auto=False, thread_id="T-rej")
    assert env["interrupt"]["type"] == "accept"
    env = service.resume("T-rej", "The steps did not work, the router is managed by IT", app=app)
    assert env["status"] == "needs_input" and env["interrupt"]["type"] == "accept"  # re-proposed
    assert llm.calls.count("You are a support investigation agent."[:40]) >= 2 or len(llm.calls) > 8
    env = service.resume("T-rej", "accept", app=app)
    assert env["status"] == "completed" and env["accepted"] and env["indexed"]


def test_rejected_case_is_not_indexed(tmp_path, install_llm):
    install_llm()
    app = app_for(tmp_path)
    run(app, auto=False, thread_id="T-rej2")
    for _ in range(4):  # keep rejecting until the loop bound is reached
        env = service.resume("T-rej2", "still wrong", app=app)
        if env["status"] == "completed":
            break
    assert env["status"] == "completed" and not env["accepted"] and not env["indexed"]


def test_long_term_memory_reused_across_threads(tmp_path, install_llm):
    install_llm()
    app = app_for(tmp_path)
    first = run(app, customer="acme", thread_id="T-a")
    assert first["stored_memories"]
    llm2 = install_llm()
    second = run(app, customer="acme", thread_id="T-b")
    # second conversation saw the first (history + memory) at intake
    intake = [e for e in cases.case_events(second["case_id"]) if e["kind"] == "intake"][0]
    assert intake["payload"]["memory_items"] >= 1 and intake["payload"]["history"] >= 1
    assert llm2.calls
    # a different customer gets nothing
    other = run(app, customer="someone-else", thread_id="T-c")
    oi = [e for e in cases.case_events(other["case_id"]) if e["kind"] == "intake"][0]
    assert oi["payload"]["memory_items"] == 0 and oi["payload"]["history"] == 0


def test_newly_resolved_case_is_retrievable(tmp_path, install_llm):
    install_llm()
    env = run(app_for(tmp_path), customer="acme")
    assert env["indexed"]
    hits = hybrid_search("VPN keeps dropping disconnects every few minutes shared drive", k=8)
    live = [h for h in hits if h.source == "live"]
    assert live and live[0].ticket_id == env["case_id"]
    with db() as c:
        assert c.execute("SELECT COUNT(*) FROM tickets WHERE source='live'").fetchone()[0] == 1


def test_tools_are_model_selected_not_fixed(tmp_path, install_llm):
    install_llm(use_tools=False)
    env = run(app_for(tmp_path))
    assert "tool_call" not in event_kinds(env["case_id"])  # model chose not to call tools
    assert env["status"] == "completed"


def scripted_reader(lines):
    it = iter(lines)

    def read(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def test_chat_session_multi_turn_clarify_accept_and_second_ticket(tmp_path, install_llm):
    """One process: ticket -> clarification -> answer -> accept -> second ticket (memory loaded)."""
    from support_agent.chat import chat_loop

    install_llm(
        assess=[
            {"route": "ask_user", "hypothesis": "h", "question": "Which device is affected?"},
            {"route": "resolve", "hypothesis": "VPN client outdated", "reason": "ok"},
        ]
    )
    app = app_for(tmp_path)
    done = chat_loop(
        "chat-cust",
        read=scripted_reader(
            [
                VAGUE["body"],  # ticket 1
                "A company laptop on the VPN",  # answers the clarification
                "accept",  # accepts the outcome
                TICKET["body"],  # ticket 2 in the same session
                "accept",
                "/exit",
            ]
        ),
        app=app,
    )
    assert done == 2
    rows = cases.customer_cases("chat-cust", limit=10)
    assert len(rows) == 2 and all(r["status"] == "INDEXED" for r in rows)
    first = next(r for r in rows if r["subject"].startswith("Hi, it is not working"))
    clar = [e for e in cases.case_events(first["case_id"]) if e["kind"] == "clarification"]
    assert len(clar) == 1
    assert clar[0]["payload"]["question"] == "Which device is affected?"
    assert clar[0]["payload"]["answer"] == "A company laptop on the VPN"


def test_chat_rejection_feedback_and_exit_midway(tmp_path, install_llm):
    from support_agent.chat import chat_loop

    install_llm()
    app = app_for(tmp_path)
    done = chat_loop(
        "chat-cust-2",
        read=scripted_reader([TICKET["body"], "not helpful, please check the router", "/exit"]),
        app=app,
    )
    assert done == 0  # left while the revised outcome was awaiting acceptance
    row = cases.customer_cases("chat-cust-2", limit=5)[0]
    assert row["status"] != "INDEXED"  # never accepted -> never added to trusted knowledge
    kinds = event_kinds(row["case_id"])
    assert kinds.count("assess") >= 2  # rejection feedback triggered a second investigation
