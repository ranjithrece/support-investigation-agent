"""Prompt builders. Only the skill catalogue + selected skill bodies are ever injected."""

from __future__ import annotations

from typing import Any

from support_agent.graph.state import AgentState
from support_agent.skills_loader import catalogue

GROUNDING_RULES = (
    "Rules: the ONLY knowledge source is the historical support tickets provided or retrieved "
    "via tools. Never use outside knowledge, never invent product names, URLs, prices, "
    "phone numbers or timelines. Cite ticket IDs. Compare several cases; do not copy one blindly."
)


def fmt_evidence(items: list[dict], n: int = 6, with_answer: bool = True) -> str:
    if not items:
        return "(no evidence retrieved)"
    lines = []
    for e in items[:n]:
        line = (
            f"[{e['ticket_id']}|{e.get('source')}|{e.get('queue')}|{e.get('type')}|"
            f"{e.get('priority')}|sim={e.get('similarity')}] PROBLEM: {e.get('summary', '')[:220]}"
        )
        if with_answer:
            line += f" || ANSWER: {e.get('answer_snippet', '')[:320]}"
        lines.append(line)
    return "\n".join(lines)


def fmt_memory(mem: list[dict]) -> str:
    return "\n".join(f"- ({m['kind']}) {m['text']}" for m in mem) or "(none)"


def fmt_history(hist: list[dict]) -> str:
    return (
        "\n".join(
            f"- {h['case_id']} [{h['status']}] {h['subject']} -> {(h.get('resolution') or 'n/a')[:120]}"
            for h in hist[:5]
        )
        or "(none)"
    )


def fmt_clarifications(cl: list[dict]) -> str:
    return "\n".join(f"Q: {c['question']}\nA: {c['answer']}" for c in cl) or "(none)"


def ticket_block(s: AgentState) -> str:
    return f"Subject: {s['subject']}\nBody: {s['body']}"


def triage_prompts(s: AgentState, peek: list[dict], skills_text: str) -> tuple[str, str]:
    system = (
        "You triage customer support tickets.\n"
        f"{GROUNDING_RULES}\n\nAvailable skills (select 1-2 names):\n{catalogue()}\n\n{skills_text}"
    )
    user = (
        f"TICKET\n{ticket_block(s)}\n\nCUSTOMER MEMORY\n{fmt_memory(s.get('customer_memory', []))}\n\n"
        f"CUSTOMER HISTORY\n{fmt_history(s.get('customer_history', []))}\n\n"
        f"SIMILAR HISTORICAL TICKETS (preview)\n{fmt_evidence(peek, 5, with_answer=False)}\n\n"
        "Classify the ticket (queue and type should be values seen in the similar tickets; "
        "priority one of low/medium/high/critical), pick 1-2 skills from the catalogue "
        "(always include incident-investigation unless the ticket is trivial), and list any "
        "missing information that would change the resolution."
    )
    return system, user


def investigate_system(s: AgentState, skills_text: str) -> str:
    return (
        "You are a support investigation agent. Investigate the ticket using the tools; you decide "
        "which tools to call and when. Call `load_skill` if you need instructions from a skill "
        "that is not loaded.\n"
        f"{GROUNDING_RULES}\n\nSkill catalogue:\n{catalogue()}\n\n{skills_text}\n\n"
        "The RETRIEVED EVIDENCE in the user message is only a first-pass keyword/semantic match. "
        "Form a hypothesis, then verify it: call `search_similar_tickets` with a query targeted at "
        "that hypothesis (and filters when triage is confident); call `get_ticket_by_id` to read the "
        "most promising case in full when answers differ; call `get_customer_history` if the "
        "customer may have raised this before. Skip a tool only if it cannot add information.\n"
        "When done, reply WITHOUT tool calls with a concise FINDINGS summary: hypothesis, the "
        "consensus steps seen across cases (with ticket IDs), conflicts between cases, and what "
        "information is still missing (if any)."
    )


def investigate_user(s: AgentState) -> str:
    ea = s.get("evidence_assessment", {})
    parts = [
        f"TICKET\n{ticket_block(s)}",
        f"CLASSIFICATION (triage): {s.get('classification')}",
        f"CUSTOMER MEMORY\n{fmt_memory(s.get('customer_memory', []))}",
        f"CLARIFICATIONS SO FAR\n{fmt_clarifications(s.get('clarifications', []))}",
        f"RETRIEVED EVIDENCE\n{fmt_evidence(s.get('evidence', []), 6)}",
        f"EVIDENCE COMPARISON: agreement={ea.get('agreement')}, "
        f"top_similarity={ea.get('top_similarity')}, conflicts={ea.get('conflicts')}, "
        f"gaps={ea.get('coverage_gaps')}",
        f"QUEUE STATS: {s.get('queue_stats', 'n/a')}",
    ]
    if s.get("feedback"):
        parts.append(f"REVIEWER/USER FEEDBACK TO ADDRESS: {s['feedback']}")
    if s.get("findings"):
        parts.append(f"PREVIOUS FINDINGS: {s['findings'][:800]}")
    return "\n\n".join(parts)


def assess_prompts(s: AgentState, skills_text: str) -> tuple[str, str]:
    system = (
        "You decide the next step of a support investigation.\n"
        "route=resolve: evidence (>=2 consistent cases) supports a concrete resolution.\n"
        "route=ask_user: ONLY when evidence agreement is low/mixed or the ticket is too vague to "
        "search, AND one specific missing fact would change the resolution (give ONE focused "
        "question). Never ask when evidence agreement is high, never repeat or rephrase a "
        "question already answered above.\n"
        "route=retrieve_more: you have a hypothesis but need targeted evidence (give followup_query).\n"
        "route=escalate: no relevant evidence, unresolved conflict, or high-risk issue.\n"
        f"{GROUNDING_RULES}\n\n{skills_text}"
    )
    ea = s.get("evidence_assessment", {})
    user = (
        f"TICKET\n{ticket_block(s)}\n\nCLASSIFICATION: {s.get('classification')}\n"
        f"CLARIFICATIONS SO FAR ({len(s.get('clarifications', []))}):\n"
        f"{fmt_clarifications(s.get('clarifications', []))}\n\n"
        f"INVESTIGATION FINDINGS\n{s.get('findings', '')[:1500]}\n\n"
        f"EVIDENCE COMPARISON: {ea}\n\n"
        f"TOP EVIDENCE\n{fmt_evidence(s.get('evidence', []), 5)}\n\n"
        f"Investigation round {s.get('investigation_round', 0)}. Choose the route."
    )
    return system, user


def draft_prompts(s: AgentState, skills_text: str, escalate: bool) -> tuple[str, str]:
    mode = (
        "Write an ESCALATION: set escalation_required=true with an evidence-based reason, a "
        "target_queue, and a short holding message to the customer as `resolution`."
        if escalate
        else "Write the resolution. Set escalation_required=false unless evidence is truly "
        "inadequate. Cite ticket IDs in `analysis` and list them in cited_ticket_ids."
    )
    system = f"You draft the final support outcome.\n{GROUNDING_RULES}\n\n{skills_text}\n\n{mode}"
    user = (
        f"TICKET\n{ticket_block(s)}\n\nCLASSIFICATION: {s.get('classification')}\n"
        f"CUSTOMER MEMORY\n{fmt_memory(s.get('customer_memory', []))}\n\n"
        f"CLARIFICATIONS\n{fmt_clarifications(s.get('clarifications', []))}\n\n"
        f"FINDINGS\n{s.get('findings', '')[:1500]}\n\n"
        f"EVIDENCE COMPARISON: {s.get('evidence_assessment')}\n\n"
        f"EVIDENCE\n{fmt_evidence(s.get('evidence', []), 6)}"
    )
    if s.get("feedback"):
        user += f"\n\nFEEDBACK TO ADDRESS: {s['feedback']}"
    return system, user


def verify_prompts(s: AgentState, draft: dict[str, Any], cited: list[dict]) -> tuple[str, str]:
    system = (
        "You are a strict fact-checker. Check whether every concrete claim, step, and number in "
        "the RESOLUTION is supported by the EVIDENCE answers. grounding_score: 1.0 = fully "
        "supported, 0.5 = partly, 0 = unsupported. List unsupported claims verbatim (short)."
    )
    user = (
        f"EVIDENCE\n{fmt_evidence(cited, 8)}\n\nRESOLUTION\n{draft.get('resolution')}\n\n"
        f"ANALYSIS\n{draft.get('analysis')}"
    )
    return system, user


def memory_prompts(s: AgentState, resolution: str) -> tuple[str, str]:
    system = (
        "Extract durable facts about the CUSTOMER worth remembering for future, separate "
        "conversations. Allowed kinds: environment (product/stack/setup they use), pattern "
        "(recurring problem), preference (language/tone/contact), caveat (unresolved constraint). "
        "Each item: one short sentence about the customer. DO NOT include greetings, the raw "
        "ticket, one-off details, emails, phone numbers, credentials, or the generic fix. "
        "Return an empty list if nothing is durable."
    )
    user = (
        f"TICKET\n{ticket_block(s)}\n\nCLARIFICATIONS\n{fmt_clarifications(s.get('clarifications', []))}"
        f"\n\nACCEPTED RESOLUTION\n{resolution[:600]}\n\nEXISTING MEMORY\n"
        f"{fmt_memory(s.get('customer_memory', []))}"
    )
    return system, user
