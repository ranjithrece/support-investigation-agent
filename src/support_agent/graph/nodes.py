"""Graph nodes. Interrupt nodes are kept tiny because LangGraph re-runs a node from its start on resume."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from support_agent.config import get_settings
from support_agent.graph import prompts as P
from support_agent.graph.state import AgentState
from support_agent.llm import get_llm, strip_think, structured
from support_agent.memory.long_term import load_memories, save_memories
from support_agent.retrieval.compare import compare_evidence
from support_agent.retrieval.hybrid import hybrid_search
from support_agent.retrieval.index import index_live_case
from support_agent.schemas import (
    AssessOutput,
    CaseResult,
    Classification,
    DraftResolution,
    EvidenceItem,
    MemoryExtraction,
    TriageOutput,
    VerifyOutput,
)
from support_agent.scoring import compute_confidence, is_trusted
from support_agent.skills_loader import load_skills, skill_names
from support_agent.storage import cases
from support_agent.storage.db import db
from support_agent.tools.registry import INVESTIGATION_TOOLS, TOOLS_BY_NAME

log = logging.getLogger(__name__)
S = get_settings

ACCEPT_WORDS = {"accept", "accepted", "yes", "y", "ok", "okay", "approve", "approved", "true"}
MAX_QUESTIONS = 2


# --------------------------------------------------------------------------- helpers
@lru_cache
def _known_values() -> dict[str, list[str]]:
    with db() as c:
        return {
            col: [
                r[0] for r in c.execute(f"SELECT DISTINCT {col} FROM tickets WHERE source='hist'")
            ]
            for col in ("queue", "type")
        }


def _normalise_classification(cls: Classification, peek: list[dict]) -> dict[str, Any]:
    known = _known_values()
    ci = {v.lower(): v for v in known.get("queue", [])}
    queue = ci.get(cls.queue.strip().lower())
    if queue is None:  # model invented a queue -> fall back to what similar tickets say
        queues = [p["queue"] for p in peek if p.get("queue")]
        queue = max(set(queues), key=queues.count) if queues else cls.queue
    ti = {v.lower(): v for v in known.get("type", ["Incident", "Request", "Problem", "Change"])}
    type_ = ti.get(cls.type.strip().lower(), "Incident")
    prio = cls.priority.strip().lower()
    if prio not in {"low", "medium", "high", "critical"}:
        prio = "medium"
    return {"queue": queue, "type": type_, "priority": prio, "tags": cls.tags[:6]}


def _repeats(question: str, prior: list[str]) -> bool:
    """True when `question` is (almost) a rephrasing of an earlier clarification question."""
    q = set(re.findall(r"[a-z0-9]{4,}", question.lower()))
    for p in prior:
        w = set(re.findall(r"[a-z0-9]{4,}", p.lower()))
        if q and w and len(q & w) / min(len(q), len(w)) >= 0.5:
            return True
    return False


_STOPW = {
    "please",
    "your",
    "that",
    "this",
    "with",
    "have",
    "will",
    "from",
    "about",
    "would",
    "could",
}


def lexical_grounding(text: str, evidence: list[dict]) -> float:
    """Fraction of the resolution's content words that appear in the evidence (fallback verifier)."""
    words = {w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOPW}
    if not words:
        return 0.0
    corpus = " ".join(
        f"{e.get('summary', '')} {e.get('answer_snippet', '')}" for e in evidence
    ).lower()
    have = set(re.findall(r"[a-z]{5,}", corpus))
    return round(min(1.0, (len(words & have) / len(words)) / 0.6), 2)


def _evidence_objs(items: list[dict]) -> list[EvidenceItem]:
    return [EvidenceItem(**i) for i in items]


def _event(state: AgentState, kind: str, payload: dict[str, Any] | None = None) -> None:
    if state.get("case_id"):
        cases.log_event(state["case_id"], kind, payload)


# --------------------------------------------------------------------------- nodes
def intake(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Persist the ticket as an OPEN case; load customer long-term memory and case history."""
    thread_id = config["configurable"]["thread_id"]
    case_id = cases.create_open_case(
        state["customer_id"], thread_id, state["subject"], state["body"]
    )
    mem = load_memories(state["customer_id"], f"{state['subject']} {state['body']}")
    hist = cases.customer_cases(state["customer_id"], limit=5, exclude=case_id)
    _event(state | {"case_id": case_id}, "intake", {"memory_items": len(mem), "history": len(hist)})
    return {
        "case_id": case_id,
        "thread_id": thread_id,
        "customer_memory": mem,
        "customer_history": [
            {k: h.get(k) for k in ("case_id", "status", "subject", "resolution", "queue", "type")}
            for h in hist
        ],
        "messages": [HumanMessage(content=f"[ticket] {state['subject']}\n{state['body']}")],
        "investigation_round": 0,
        "verify_attempts": 0,
        "rejections": 0,
        "clarifications": [],
        "evidence": [],
    }


def triage(state: AgentState) -> dict[str, Any]:
    peek = [e.model_dump() for e in hybrid_search(f"{state['subject']}\n{state['body']}", k=5)]
    system, user = P.triage_prompts(state, peek, load_skills(["ticket-triage"]))
    out = structured(TriageOutput, system, user)
    cls = _normalise_classification(out.classification, peek)
    skills = [s for s in out.skills if s in skill_names()][:2] or ["incident-investigation"]
    if "incident-investigation" not in skills:
        skills.append("incident-investigation")
    cases.update_classification(
        state["case_id"], cls["queue"], cls["type"], cls["priority"], cls["tags"]
    )
    _event(state, "triage", {"classification": cls, "skills": skills, "missing": out.missing_info})
    return {"classification": cls, "selected_skills": skills, "missing_info": out.missing_info}


# parallel retrieval fan-out (3 independent branches, merged by the `evidence` reducer)
def retrieve_filtered(state: AgentState) -> dict[str, Any]:
    cls = state.get("classification", {})
    items = hybrid_search(
        f"{state['subject']}\n{state['body']}",
        queue=cls.get("queue"),
        type_=cls.get("type"),
        tags=cls.get("tags"),
        k=S().retrieval_k,
    )
    return {"evidence": [i.model_dump() for i in items]}


def retrieve_broad(state: AgentState) -> dict[str, Any]:
    """Unfiltered search: catches cases whose queue differs from the triage guess."""
    items = hybrid_search(f"{state['subject']}\n{state['body']}", k=S().retrieval_k)
    return {"evidence": [i.model_dump() for i in items]}


def retrieve_stats(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    cls = state.get("classification", {})
    tool = TOOLS_BY_NAME["get_queue_tag_statistics"]
    txt = tool.invoke({"queue": cls.get("queue"), "type": cls.get("type")}, config=config)
    return {"queue_stats": str(txt)[:600]}


def retrieve_followup(state: AgentState) -> dict[str, Any]:
    """Hypothesis-driven targeted retrieval (also used after a clarification answer)."""
    cls = state.get("classification", {})
    q = state.get("followup_query") or f"{state['subject']} {state.get('hypothesis', '')}"
    items = hybrid_search(q, queue=cls.get("queue"), k=S().retrieval_k)
    broad = hybrid_search(q, k=4)
    _event(state, "followup_retrieval", {"query": q, "n": len(items) + len(broad)})
    return {"evidence": [i.model_dump() for i in items + broad], "followup_query": None}


def compare(state: AgentState) -> dict[str, Any]:
    assessment = compare_evidence(_evidence_objs(state.get("evidence", [])))
    _event(state, "compare_evidence", assessment.model_dump())
    return {"evidence_assessment": assessment.model_dump()}


def investigate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Model-driven tool loop: the LLM decides which tools to call (bounded)."""
    s = S()
    llm = get_llm().bind_tools(INVESTIGATION_TOOLS)
    msgs: list[Any] = [
        SystemMessage(
            content=P.investigate_system(state, load_skills(state.get("selected_skills", [])))
        ),
        HumanMessage(content=P.investigate_user(state)),
    ]
    new_evidence: list[dict] = []
    ai: AIMessage | None = None
    for _ in range(s.max_tool_steps):
        ai = llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            tool = TOOLS_BY_NAME.get(tc["name"])
            if tool is None:
                msgs.append(
                    ToolMessage(content=f"Unknown tool {tc['name']}", tool_call_id=tc["id"])
                )
                continue
            _event(state, "tool_call", {"name": tc["name"], "args": tc["args"]})
            try:
                tm = tool.invoke(tc, config=config)
                content, artifact = tm.content, getattr(tm, "artifact", None)
            except Exception as e:  # tool failure is fed back to the model, not raised
                log.warning("tool %s failed: %s", tc["name"], e)
                content, artifact = f"Tool error: {type(e).__name__}", None
            if isinstance(artifact, list):
                new_evidence.extend(artifact)
            msgs.append(ToolMessage(content=str(content)[:4000], tool_call_id=tc["id"]))
    else:  # tool budget exhausted -> force a summary
        msgs.append(HumanMessage(content="Tool budget used. Give your FINDINGS summary now."))
        ai = get_llm().invoke(msgs)

    findings = strip_think(str(ai.content)) if ai is not None else ""
    if not findings:
        findings = "No findings produced; rely on retrieved evidence."
    case = cases.get_case(state["case_id"]) or {}
    cls = dict(state.get("classification", {}))
    if case.get("queue"):  # the model may have called update_classification
        cls = {
            "queue": case["queue"],
            "type": case["type"],
            "priority": case["priority"],
            "tags": case.get("tags", cls.get("tags", [])),
        }
    return {
        "findings": findings,
        "evidence": new_evidence,
        "classification": cls,
        "investigation_round": state.get("investigation_round", 0) + 1,
        "messages": [AIMessage(content=f"[findings] {findings[:1200]}")],
    }


def assess(state: AgentState) -> dict[str, Any]:
    """Decide the next route; deterministic guardrails bound the loops."""
    s = S()
    ea = state.get("evidence_assessment", {})
    system, user = P.assess_prompts(state, load_skills(["clarification", "escalation"]))
    out = structured(AssessOutput, system, user)
    route = out.route
    n_q = len(state.get("clarifications", []))
    rnd = state.get("investigation_round", 0)
    weak = ea.get("agreement") == "low" or not state.get("evidence")

    prior = [c["question"] for c in state.get("clarifications", [])]
    if route == "ask_user" and (
        n_q >= MAX_QUESTIONS
        or not out.question
        or _repeats(out.question, prior)
        or ea.get("agreement") == "high"  # PDF: ask only when evidence is insufficient/conflicting
    ):
        route = "escalate" if weak else "resolve"
    if route == "retrieve_more" and (rnd >= s.max_investigation_rounds or not out.followup_query):
        route = "escalate" if weak else "resolve"
    if route == "resolve" and not state.get("evidence"):
        route = "escalate"
    if state.get("feedback") and route == "ask_user":
        route = "resolve"
    _event(state, "assess", {"llm_route": out.route, "route": route, "reason": out.reason})
    return {
        "route": route,
        "hypothesis": out.hypothesis,
        "pending_question": out.question if route == "ask_user" else None,
        "followup_query": out.followup_query if route == "retrieve_more" else None,
    }


def ask_user(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop: pause the graph until the user answers."""
    question = state.get("pending_question") or "Could you share more details about the issue?"
    cases.set_status(state["case_id"], "AWAITING_USER")
    answer = interrupt({"type": "clarification", "case_id": state["case_id"], "question": question})
    answer = str(answer).strip()
    cases.set_status(state["case_id"], "OPEN")
    cases.log_event(
        state["case_id"], "clarification", {"question": question, "answer": answer[:300]}
    )
    return {
        "clarifications": [{"question": question, "answer": answer}],
        "messages": [AIMessage(content=f"[question] {question}"), HumanMessage(content=answer)],
        "pending_question": None,
        "followup_query": f"{state['subject']} {answer}",
    }


def _draft(state: AgentState, escalate: bool) -> dict[str, Any]:
    skills = ["customer-response"] + (["escalation"] if escalate else [])
    system, user = P.draft_prompts(state, load_skills(skills), escalate)
    d = structured(DraftResolution, system, user)
    if escalate:
        d.escalation_required = True
    return {"draft": d.model_dump(), "feedback": None}


def resolve(state: AgentState) -> dict[str, Any]:
    return _draft(state, escalate=False)


def escalate(state: AgentState) -> dict[str, Any]:
    return _draft(state, escalate=True)


def verify(state: AgentState) -> dict[str, Any]:
    """Evidence-check the draft; compute confidence; persist the proposal."""
    s = S()
    draft = state["draft"]
    by_id = {e["ticket_id"]: e for e in state.get("evidence", [])}
    cited = [by_id[i] for i in draft.get("cited_ticket_ids", []) if i in by_id]
    support = cited or state.get("evidence", [])[:6]
    escalated = bool(draft.get("escalation_required"))

    if escalated:
        grounding, unsupported = 1.0, []
    else:
        system, user = P.verify_prompts(state, draft, support)
        try:
            v = structured(VerifyOutput, system, user, retries=3)
            grounding, unsupported = v.grounding_score, v.unsupported_claims
        except RuntimeError as e:  # LLM verifier unavailable -> deterministic lexical check
            log.warning("verifier failed, using lexical fallback: %s", e)
            grounding, unsupported = lexical_grounding(draft["resolution"], support), []
            _event(state, "verifier_fallback", {"score": grounding})
    grounded = escalated or (grounding >= 0.6 and len(unsupported) <= 2)

    cls = Classification(**state["classification"])
    conf = compute_confidence(
        state.get("evidence_assessment", {}),
        grounding,
        state["classification"],
        state.get("evidence", []),
        escalated,
    )
    result = CaseResult(
        case_id=state["case_id"],
        classification=cls,
        evidence=_evidence_objs((cited or state.get("evidence", [])[:4])[:6]),
        analysis=draft["analysis"],
        resolution=draft["resolution"],
        escalation={  # type: ignore[arg-type]
            "required": escalated,
            "reason": draft.get("escalation_reason") if escalated else None,
            "target_queue": (draft.get("target_queue") or cls.queue) if escalated else None,
        },
        confidence=conf,
        verification={  # type: ignore[arg-type]
            "grounded": grounded,
            "grounding_score": round(grounding, 2),
            "unsupported_claims": unsupported,
        },
    ).model_dump()
    attempts = state.get("verify_attempts", 0) + 1
    cases.save_proposal(state["case_id"], result)
    _event(
        state,
        "verify",
        {
            "grounded": grounded,
            "score": grounding,
            "attempt": attempts,
            "confidence": conf,
            "unsupported": unsupported[:4],
        },
    )
    update: dict[str, Any] = {
        "result": result,
        "verification": result["verification"],
        "verify_attempts": attempts,
    }
    if not grounded and attempts <= s.max_verify_retries:
        update["feedback"] = "Unsupported claims to remove or support: " + "; ".join(
            unsupported[:4]
        )
    return update


def await_accept(state: AgentState) -> dict[str, Any]:
    """Human acceptance gate (skipped with auto_accept)."""
    if state.get("auto_accept"):
        return {"user_accepted": True}
    reply = interrupt({"type": "accept", "case_id": state["case_id"], "result": state["result"]})
    if isinstance(reply, dict):
        accepted, feedback = bool(reply.get("accepted")), str(reply.get("feedback") or "")
    else:
        text = str(reply).strip()
        accepted = text.lower() in ACCEPT_WORDS
        feedback = "" if accepted else text
    return {
        "user_accepted": accepted,
        "feedback": None
        if accepted
        else (feedback or "The user rejected the proposed resolution."),
        "rejections": state.get("rejections", 0) + (0 if accepted else 1),
        "verify_attempts": 0 if not accepted else state.get("verify_attempts", 0),
    }


def persist(state: AgentState) -> dict[str, Any]:
    """Store the final case; write worthwhile long-term memory (accepted cases only)."""
    result = state["result"]
    accepted = bool(state.get("user_accepted"))
    escalated = result["escalation"]["required"]
    cases.finalize(state["case_id"], accepted, escalated and accepted)
    stored: list[dict] = []
    if accepted:
        items: list[dict[str, str]] = []
        try:
            sys_, usr = P.memory_prompts(state, result["resolution"])
            ext = structured(MemoryExtraction, sys_, usr, retries=1)
            items = [m.model_dump() for m in ext.items][:5]
        except RuntimeError:
            log.warning("memory extraction failed; storing outcome only")
        if not escalated:
            items.append(
                {
                    "kind": "outcome",
                    "text": f"Past accepted case: {state['subject'][:90]} (queue {result['classification']['queue']})",
                }
            )
        said = " ".join(
            [state["subject"], state["body"]]
            + [c["answer"] for c in state.get("clarifications", [])]
        )
        stored = save_memories(state["customer_id"], items, state["case_id"], source_text=said)
    _event(
        state, "persist", {"accepted": accepted, "escalated": escalated, "memories": len(stored)}
    )
    return {"stored_memories": stored}


def index_case(state: AgentState) -> dict[str, Any]:
    """Make a trusted resolved case searchable for future tickets."""
    case = cases.get_case(state["case_id"])
    assert case is not None
    index_live_case(case)
    cases.mark_indexed(state["case_id"])
    _event(state, "indexed", {"case_id": state["case_id"]})
    return {"indexed": True}


def trusted(state: AgentState) -> bool:
    r = state.get("result") or {}
    return is_trusted(
        bool(state.get("user_accepted")),
        bool(r.get("escalation", {}).get("required")),
        bool(r.get("verification", {}).get("grounded")),
        float(r.get("confidence", 0)),
        S().trust_confidence_threshold,
    )
