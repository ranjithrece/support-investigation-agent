"""LangGraph state schema with reducers (parallel retrieval branches merge into `evidence`)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_evidence(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Union by ticket_id keeping the best similarity; sorted best-first."""
    best: dict[str, dict] = {}
    for item in (left or []) + (right or []):
        cur = best.get(item["ticket_id"])
        if cur is None or item.get("similarity", 0) > cur.get("similarity", 0):
            best[item["ticket_id"]] = item
    return sorted(best.values(), key=lambda e: e.get("similarity", 0), reverse=True)


class AgentState(TypedDict, total=False):
    # identity / input
    case_id: str
    customer_id: str
    thread_id: str
    subject: str
    body: str
    auto_accept: bool

    # short-term (thread) memory
    messages: Annotated[list[AnyMessage], add_messages]
    clarifications: Annotated[list[dict], operator.add]  # {"question","answer"}

    # long-term memory loaded at intake
    customer_memory: list[dict]
    customer_history: list[dict]

    # triage
    classification: dict[str, Any]
    selected_skills: list[str]
    missing_info: list[str]

    # evidence
    evidence: Annotated[list[dict], merge_evidence]
    evidence_assessment: dict[str, Any]
    queue_stats: str
    hypothesis: str
    findings: str
    followup_query: str | None

    # control flow
    investigation_round: int
    verify_attempts: int
    rejections: int
    route: Literal["resolve", "ask_user", "retrieve_more", "escalate"]
    pending_question: str | None
    feedback: str | None  # verifier / user feedback driving a re-investigation

    # drafting / output
    draft: dict[str, Any]
    verification: dict[str, Any]
    result: dict[str, Any]
    user_accepted: bool | None
    indexed: bool
    stored_memories: list[dict]
