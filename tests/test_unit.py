"""Unit tests for pure logic: comparison, confidence, memory gate, skills, FTS sanitising."""

from __future__ import annotations

from support_agent.graph.state import merge_evidence
from support_agent.ingest.loader import clean_text
from support_agent.memory.long_term import is_write_worthy, load_memories, save_memories, scrub
from support_agent.retrieval.compare import compare_evidence
from support_agent.retrieval.hybrid import hybrid_search
from support_agent.schemas import EvidenceItem
from support_agent.scoring import compute_confidence, is_trusted
from support_agent.skills_loader import catalogue, load_skill_body, load_skills, skill_names
from support_agent.storage.db import fts_query


def ev(i, queue="Technical Support", type_="Incident", sim=0.8, ans="update client restart router"):
    return EvidenceItem(
        ticket_id=f"H-{i}", queue=queue, type=type_, similarity=sim, answer_snippet=ans
    )


def test_compare_high_agreement():
    a = compare_evidence([ev(i) for i in range(4)])
    assert a.agreement == "high" and a.conflicts == [] and a.dominant_queue == "Technical Support"


def test_compare_detects_conflict_and_low_similarity():
    items = [
        ev(1, "Billing and Payments", "Request", 0.3, "refund card"),
        ev(2, "Technical Support", "Incident", 0.3, "reinstall driver firmware"),
        ev(3, "Customer Service", "Problem", 0.3, "escalate account manager"),
    ]
    a = compare_evidence(items)
    assert a.agreement == "low" and a.conflicts and a.coverage_gaps


def test_compare_empty():
    a = compare_evidence([])
    assert a.agreement == "low" and a.coverage_gaps


def test_confidence_ordering_and_escalation_cap():
    good = compare_evidence([ev(i) for i in range(4)]).model_dump()
    bad = compare_evidence([ev(1, sim=0.2, ans="x")]).model_dump()
    cls = {"queue": "Technical Support"}
    e = [ev(i).model_dump() for i in range(4)]
    hi = compute_confidence(good, 0.95, cls, e)
    lo = compute_confidence(bad, 0.4, cls, e[:1])
    assert hi > lo and 0 <= lo <= hi <= 1
    assert compute_confidence(good, 1.0, cls, e, escalated=True) <= 0.4


def test_trust_gate():
    assert is_trusted(True, False, True, 0.8, 0.7)
    assert not is_trusted(False, False, True, 0.9, 0.7)
    assert not is_trusted(True, True, True, 0.9, 0.7)
    assert not is_trusted(True, False, False, 0.9, 0.7)
    assert not is_trusted(True, False, True, 0.5, 0.7)


def test_memory_gate_filters_junk_and_pii():
    assert not is_write_worthy("Hi team", "pattern")
    assert not is_write_worthy("Dear support, I am writing about a thing", "pattern")
    assert not is_write_worthy(
        "Contact me at john@example.com please now", "preference"
    )  # after scrub
    assert is_write_worthy("Uses on-prem Okta SSO with Windows laptops", "environment")
    assert "[redacted]" in scrub("mail john@example.com or +1 415 555 0100")


def test_memory_save_dedupe_and_load():
    stored = save_memories(
        "c1",
        [
            {"kind": "environment", "text": "Customer runs a Cisco VPN client on macOS laptops"},
            {
                "kind": "environment",
                "text": "Customer runs Cisco VPN client on macOS laptops",
            },  # near-duplicate
            {"kind": "pattern", "text": "ok"},  # too short
            {"kind": "preference", "text": "Reach me at a@b.com"},  # PII -> rejected
        ],
        "C-1",
    )
    assert len(stored) == 1
    mem = load_memories("c1", "vpn macos")
    assert len(mem) == 1 and "Cisco" in mem[0]["text"]
    assert load_memories("other", "vpn") == []  # customer isolation


def test_skills_dynamic_loading():
    assert len(skill_names()) >= 3
    cat = catalogue()
    assert "ticket-triage" in cat and "Skill:" not in cat  # catalogue holds no full bodies
    assert "Skill: escalation" in load_skill_body("escalation")
    assert load_skills(["nope"]) == ""


def test_fts_query_is_sanitised():
    q = fts_query('vpn" OR 1=1; DROP TABLE tickets --')
    assert "DROP" not in q.upper() or '"drop"' in q  # tokens are quoted, never raw SQL/FTS syntax
    assert fts_query("a b") == ""


def test_merge_evidence_dedupes_keeps_best():
    out = merge_evidence(
        [{"ticket_id": "a", "similarity": 0.4}],
        [{"ticket_id": "a", "similarity": 0.9}, {"ticket_id": "b", "similarity": 0.5}],
    )
    assert [e["ticket_id"] for e in out] == ["a", "b"] and out[0]["similarity"] == 0.9


def test_clean_text():
    assert clean_text("a\\nb   c\r\n\n\n\nd") == "a\nb c\n\nd"
    assert clean_text(float("nan")) == ""


def test_hybrid_search_finds_relevant_and_filters():
    hits = hybrid_search("VPN disconnects shared drive", k=3)
    assert hits and all("VPN" in h.summary for h in hits)
    only_bill = hybrid_search("refund invoice", queue="Billing and Payments", k=5)
    assert only_bill and all(h.queue == "Billing and Payments" for h in only_bill[:2])


def test_memory_must_be_grounded_in_customer_words():
    said = "We run Cisco AnyConnect on macOS laptops and lose the file server."
    stored = save_memories(
        "c2",
        [
            {"kind": "environment", "text": "Customer runs Cisco AnyConnect on macOS laptops"},
            {
                "kind": "caveat",
                "text": "The issue persists despite previous troubleshooting steps",
            },  # invented
        ],
        "C-2",
        source_text=said,
    )
    assert [m["kind"] for m in stored] == ["environment"]


def test_lexical_grounding_fallback():
    from support_agent.graph.nodes import lexical_grounding

    ev = [
        {
            "summary": "VPN disconnects",
            "answer_snippet": "Update the client, restart the router and re-authenticate",
        }
    ]
    assert lexical_grounding("Update the client and restart the router", ev) >= 0.8
    assert lexical_grounding("Reformat your spaceship quantum drive immediately", ev) < 0.3


def test_preflight_reports_missing_data_and_models(monkeypatch):
    import pytest

    from support_agent import preflight

    monkeypatch.setattr(preflight, "_ollama_models", lambda url: {"nomic-embed-text:latest"})
    with pytest.raises(preflight.PreflightError, match="ollama pull"):
        preflight.check_ollama(need_chat=True)
    preflight.check_ollama(need_chat=False)  # embed model present -> ok

    def unreachable(url):
        raise preflight.PreflightError("Cannot reach Ollama")

    monkeypatch.setattr(preflight, "_ollama_models", unreachable)
    with pytest.raises(preflight.PreflightError, match="Cannot reach Ollama"):
        preflight.check_ollama()


def test_ingest_limit_keeps_holdout_and_is_idempotent(tmp_path):
    """Regression: `agent ingest --limit N` (the documented smoke test) must work."""
    import pandas as pd

    from support_agent.ingest.loader import TAG_COLS
    from support_agent.ingest.pipeline import ingest
    from support_agent.storage.db import db

    rows = []
    for i in range(60):
        r = {
            "subject": f"subject {i}",
            "body": f"body text number {i} about vpn",
            "answer": f"answer {i}",
            "type": "Incident",
            "queue": "Technical Support" if i % 2 else "Billing and Payments",
            "priority": "high",
            "language": "en",
        }
        r.update({c: None for c in TAG_COLS})
        rows.append(r)
    csv = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    with db() as c:
        before = c.execute("SELECT count(*) FROM tickets").fetchone()[0]
    stats = ingest([csv], limit=20, holdout=10, skip_vectors=True)
    with db() as c:
        after = c.execute("SELECT count(*) FROM tickets").fetchone()[0]
        held = c.execute("SELECT count(*) FROM tickets WHERE holdout=1").fetchone()[0]
    assert stats["tickets"] >= 28  # 20 indexable + ~10 hold-out (stratified rounding)
    assert held >= 8  # hold-out rows survive --limit so `agent eval` still works
    assert after > before
    assert ingest([csv], limit=20, holdout=10, skip_vectors=True)["new_rows"] == 0  # idempotent
