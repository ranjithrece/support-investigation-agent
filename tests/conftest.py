"""Shared fixtures: isolated data dir, tiny corpus, deterministic embeddings, scripted fake LLM."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DIM = 128

CORPUS = [
    (
        "VPN keeps disconnecting",
        "My VPN connection drops every few minutes and I lose access to the shared drive.",
        "Please update the VPN client to the latest version, restart the router and re-authenticate. If the drops continue, switch the tunnel protocol in the client settings.",
        "Technical Support",
        "Incident",
        "high",
        ["VPN", "Network"],
    ),
    (
        "Frequent VPN drops",
        "VPN disconnects randomly during work hours, drops connection to the file server.",
        "Update the VPN client, restart the router and re-authenticate. Changing the tunnel protocol in the client settings also stabilises the connection.",
        "Technical Support",
        "Incident",
        "high",
        ["VPN", "Network"],
    ),
    (
        "VPN unstable on laptop",
        "The VPN disconnects repeatedly on my laptop and the shared drive becomes unreachable.",
        "Update the VPN client and restart the router. Re-authenticate and change the tunnel protocol in the settings if disconnections persist.",
        "Technical Support",
        "Incident",
        "medium",
        ["VPN"],
    ),
    (
        "VPN tunnel drops on Wi-Fi",
        "The VPN tunnel disconnects on office Wi-Fi every few minutes, shared drive unreachable.",
        "Update the VPN client, restart the router and re-authenticate. Switching the tunnel protocol in the client settings resolves repeated disconnections.",
        "Technical Support",
        "Incident",
        "high",
        ["VPN", "WiFi"],
    ),
    (
        "VPN disconnects during calls",
        "My VPN keeps disconnecting every few minutes and I lose shared drive access during calls.",
        "Update the VPN client and restart the router, then re-authenticate. Change the tunnel protocol in settings if the disconnections continue.",
        "Technical Support",
        "Incident",
        "medium",
        ["VPN"],
    ),
    (
        "VPN session timeouts",
        "VPN session times out and disconnects, the shared drive disappears every few minutes.",
        "Update the VPN client, restart the router and re-authenticate. A different tunnel protocol in the client settings prevents the timeouts.",
        "Technical Support",
        "Incident",
        "high",
        ["VPN", "Network"],
    ),
    (
        "Invoice charged twice",
        "I was billed twice for my subscription this month and need a refund.",
        "We have reviewed the invoice. The duplicate charge will be refunded to the original payment method within five business days.",
        "Billing and Payments",
        "Request",
        "medium",
        ["Billing", "Refund"],
    ),
    (
        "Refund not received",
        "My refund for the duplicate invoice has not arrived yet.",
        "The refund was issued to the original payment method. Please allow five business days for the bank to post it.",
        "Billing and Payments",
        "Request",
        "low",
        ["Billing", "Refund"],
    ),
    (
        "Cannot log in to portal",
        "Password reset email never arrives and I cannot log in to the portal.",
        "Check the spam folder, then request a new password reset link. If it still fails, clear browser cache and try a different browser.",
        "Customer Service",
        "Incident",
        "medium",
        ["Login", "Password"],
    ),
    (
        "Account locked after attempts",
        "My account is locked after too many login attempts.",
        "The account lock expires automatically after thirty minutes. Afterwards use the password reset link to set a new password.",
        "Customer Service",
        "Incident",
        "medium",
        ["Login", "Account"],
    ),
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def fake_vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for t in _tokens(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Fresh data dir + deterministic embeddings for every test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from support_agent import config

    config.reset_settings_cache()
    from support_agent.graph import nodes
    from support_agent.retrieval import embeddings as emb
    from support_agent.retrieval import hybrid, index

    monkeypatch.setattr(emb, "embed_documents", lambda texts: [fake_vec(t) for t in texts])
    monkeypatch.setattr(emb, "embed_query", fake_vec)
    monkeypatch.setattr(hybrid, "embed_query", fake_vec)
    monkeypatch.setattr(index, "embed_documents", lambda texts: [fake_vec(t) for t in texts])
    nodes._known_values.cache_clear()

    from support_agent.storage.db import db, init_db

    init_db()
    col = emb.collection()
    with db() as c:
        for n, (subj, body, ans, queue, typ, prio, tags) in enumerate(CORPUS, start=1):
            tid = f"H-{n:06d}"
            c.execute(
                "INSERT INTO tickets(ticket_id, source, subject, body, answer, queue, type, priority,"
                " language, tags_json, holdout, indexed) VALUES (?,?,?,?,?,?,?,?,?,?,0,1)",
                (tid, "hist", subj, body, ans, queue, typ, prio, "en", json.dumps(tags)),
            )
            c.execute(
                "INSERT INTO tickets_fts(ticket_id, subject, body, tags) VALUES (?,?,?,?)",
                (tid, subj, body, " ".join(tags)),
            )
            col.upsert(
                ids=[tid],
                embeddings=[fake_vec(f"{subj}\n{body}")],
                documents=[body],
                metadatas=[
                    {
                        "queue": queue,
                        "type": typ,
                        "priority": prio,
                        "tags": "|".join(tags),
                        "source": "hist",
                    }
                ],
            )
    yield
    from support_agent.llm import set_llm_override

    set_llm_override(None)
    config.reset_settings_cache()


class ScriptedLLM(BaseChatModel):
    """Fake chat model: answers each graph node's prompt from a script (no network)."""

    script: dict
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw):  # noqa: ANN001
        system = str(messages[0].content)
        self.calls.append(system[:40])
        s = self.script
        if system.startswith("You triage"):
            out = AIMessage(content=json.dumps(s["triage"]))
        elif system.startswith("You are a support investigation agent"):
            had_tool = any(getattr(m, "type", "") == "tool" for m in messages)
            if s.get("use_tools", True) and not had_tool:
                out = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_similar_tickets",
                            "id": "call1",
                            "args": {"query": s.get("tool_query", "VPN disconnects shared drive")},
                        }
                    ],
                )
            else:
                out = AIMessage(
                    content=s.get(
                        "findings", "Consensus: update client, restart router (H-000001, H-000002)."
                    )
                )
        elif system.startswith("You decide the next step"):
            routes = s["assess"]
            r = routes.pop(0) if len(routes) > 1 else routes[0]
            out = AIMessage(content=json.dumps(r))
        elif system.startswith("You draft the final"):
            drafts = s["draft"]
            d = drafts.pop(0) if len(drafts) > 1 else drafts[0]
            out = AIMessage(content=json.dumps(d))
        elif system.startswith("You are a strict fact-checker"):
            vs = s["verify"]
            v = vs.pop(0) if len(vs) > 1 else vs[0]
            out = AIMessage(content=json.dumps(v))
        elif system.startswith("Extract durable facts"):
            out = AIMessage(content=json.dumps(s.get("memory", {"items": []})))
        else:  # pragma: no cover
            raise AssertionError(f"unexpected prompt: {system[:80]}")
        return ChatResult(generations=[ChatGeneration(message=out)])


def make_script(**over) -> dict:
    base = {
        "triage": {
            "classification": {
                "queue": "Technical Support",
                "type": "Incident",
                "priority": "high",
                "tags": ["VPN", "Network"],
            },
            "skills": ["ticket-triage", "incident-investigation"],
            "missing_info": [],
        },
        "assess": [
            {"route": "resolve", "hypothesis": "VPN client outdated", "reason": "consistent"}
        ],
        "draft": [
            {
                "analysis": "H-000001, H-000002 and H-000003 agree: update client, restart router.",
                "resolution": "Please update your VPN client, restart the router and re-authenticate.",
                "cited_ticket_ids": ["H-000001", "H-000002", "H-000003"],
                "escalation_required": False,
            }
        ],
        "verify": [{"grounding_score": 0.95, "unsupported_claims": []}],
        "memory": {
            "items": [
                {"kind": "environment", "text": "Customer works remotely over a corporate VPN"}
            ]
        },
    }
    base.update(over)
    return base


@pytest.fixture
def install_llm() -> Callable[..., ScriptedLLM]:
    from support_agent.llm import set_llm_override

    def _install(**over) -> ScriptedLLM:
        llm = ScriptedLLM(script=make_script(**over), calls=[])
        set_llm_override(llm)
        return llm

    return _install
