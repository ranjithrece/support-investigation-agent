"""Conditional routers + graph assembly."""

from __future__ import annotations

import sqlite3
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from support_agent.config import get_settings
from support_agent.graph import nodes as N
from support_agent.graph.state import AgentState


def route_after_assess(
    state: AgentState,
) -> Literal["resolve", "ask_user", "retrieve_followup", "escalate"]:
    return {
        "resolve": "resolve",
        "ask_user": "ask_user",
        "retrieve_more": "retrieve_followup",
        "escalate": "escalate",
    }[state["route"]]


def route_after_verify(state: AgentState) -> Literal["investigate", "escalate", "await_accept"]:
    s = get_settings()
    if state["result"]["escalation"]["required"]:
        return "await_accept"  # escalation drafts are not re-investigated (bounded loop)
    if not state["verification"]["grounded"]:
        if state.get("verify_attempts", 0) <= s.max_verify_retries:
            return "investigate"  # unsupported -> retrieve/investigate again
        return "escalate"  # still unsupported after retries -> hand to a human
    return "await_accept"


def route_after_accept(state: AgentState) -> Literal["persist", "investigate"]:
    if not state.get("user_accepted") and state.get("rejections", 0) <= 2:
        return "investigate"  # user rejected with feedback -> investigate again
    return "persist"


def route_after_persist(state: AgentState) -> Literal["index_case", "__end__"]:
    return "index_case" if N.trusted(state) else END  # type: ignore[return-value]


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [
        ("intake", N.intake),
        ("triage", N.triage),
        ("retrieve_filtered", N.retrieve_filtered),
        ("retrieve_broad", N.retrieve_broad),
        ("retrieve_stats", N.retrieve_stats),
        ("retrieve_followup", N.retrieve_followup),
        ("compare", N.compare),
        ("investigate", N.investigate),
        ("assess", N.assess),
        ("ask_user", N.ask_user),
        ("resolve", N.resolve),
        ("escalate", N.escalate),
        ("verify", N.verify),
        ("await_accept", N.await_accept),
        ("persist", N.persist),
        ("index_case", N.index_case),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "triage")
    for branch in ("retrieve_filtered", "retrieve_broad", "retrieve_stats"):  # parallel fan-out
        g.add_edge("triage", branch)
        g.add_edge(branch, "compare")
    g.add_edge("retrieve_followup", "compare")
    g.add_edge("compare", "investigate")
    g.add_edge("investigate", "assess")
    g.add_conditional_edges("assess", route_after_assess)
    g.add_edge("ask_user", "retrieve_followup")  # clarification answer -> targeted retrieval
    g.add_edge("resolve", "verify")
    g.add_edge("escalate", "verify")
    g.add_conditional_edges("verify", route_after_verify)
    g.add_conditional_edges("await_accept", route_after_accept)
    g.add_conditional_edges("persist", route_after_persist)
    g.add_edge("index_case", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer() -> SqliteSaver:
    path = get_settings().checkpoint_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def get_app():
    return build_graph(sqlite_checkpointer())
