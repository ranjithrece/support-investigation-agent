"""Demonstration of the four required paths, end to end, against the real local stack.

    uv run python scripts/demo.py --scenario all

1. normal      - clear ticket -> grounded resolution -> accepted -> persisted + indexed
2. clarify     - vague ticket -> agent pauses with a question -> resume with an answer
3. memory      - same customer opens a second ticket in a NEW thread -> prior context reused
4. growth      - a similar ticket from another customer retrieves the case resolved in (1)

Each demo uses unique customer ids, so it can be re-run safely (it adds new live cases).
"""

from __future__ import annotations

import argparse
import uuid

from rich.console import Console
from rich.rule import Rule

from support_agent import service
from support_agent.render import render_result
from support_agent.retrieval.hybrid import hybrid_search
from support_agent.storage import cases

console = Console()
RUN = uuid.uuid4().hex[:5]


def _events(case_id: str, *kinds: str) -> list[dict]:
    return [e for e in cases.case_events(case_id) if e["kind"] in kinds]


def drive(env: dict, answer: str, max_turns: int = 4) -> dict:
    """Answer any clarification (once or twice) and accept the outcome, like a cooperative user."""
    for _ in range(max_turns):
        if env["status"] != "needs_input":
            break
        i = env["interrupt"]
        if i["type"] == "clarification":
            console.print(
                f"[yellow]AGENT ASKS:[/] {i['question']}\n[green]USER ANSWERS:[/] {answer}"
            )
            env = service.resume(env["thread_id"], answer)
        else:
            env = service.resume(env["thread_id"], "accept")
    return env


def normal() -> dict:
    console.print(Rule("1. Normal resolution"))
    env = service.start_ticket(
        f"acme-{RUN}",
        "Frequent VPN disconnections",
        "Our VPN keeps disconnecting every few minutes. Remote employees lose access to the shared "
        "drive and have to reconnect repeatedly. It affects all offices, started last week after a "
        "router firmware update, and the router model and firmware are listed in our asset sheet.",
        auto_accept=True,
    )
    env = drive(env, "All offices are affected; the router firmware was updated last week.")
    assert env["status"] == "completed", env
    render_result(env["result"])
    tools = [e["payload"]["name"] for e in _events(env["case_id"], "tool_call")]
    console.print(
        f"tools chosen by the model: {tools}\nindexed for future retrieval: {env['indexed']}"
    )
    return env


def clarify() -> None:
    console.print(Rule("2. Clarification + resume (interrupt survives a fresh graph instance)"))
    env = service.start_ticket(
        f"globex-{RUN}", "It is not working", "Hi, it stopped working. Please help."
    )
    if env["status"] != "needs_input":
        console.print("[yellow]agent resolved without asking (model judged evidence sufficient)[/]")
        render_result(env["result"])
        return
    q = env["interrupt"]
    console.print(f"[yellow]AGENT ASKS ({q['type']}):[/] {q.get('question')}")
    answer = "It is our company VPN client; it disconnects every few minutes on my laptop."
    console.print(f"[green]USER ANSWERS:[/] {answer}")
    env = service.resume(env["thread_id"], answer)  # fresh get_app() => proves persistence
    env = drive(env, answer)
    render_result(env["result"])
    console.print(
        f"clarifications recorded: {[e['payload'] for e in _events(env['case_id'], 'clarification')]}"
    )


def memory() -> None:
    console.print(Rule("3. Long-term memory reuse across separate conversations"))
    cust = f"initech-{RUN}"
    first = service.start_ticket(
        cust,
        "Cannot reach file server over VPN",
        "We run Cisco AnyConnect VPN on macOS laptops and users lose the file server connection.",
        auto_accept=True,
    )
    first = drive(first, "Cisco AnyConnect on macOS laptops, all users are affected.")
    console.print("memories stored after ticket 1:", first.get("stored_memories"))
    second = service.start_ticket(
        cust, "VPN problem again", "VPN dropped again for another user today.", auto_accept=True
    )
    second = drive(second, "Same Cisco AnyConnect setup as before.")
    intake = _events(second["case_id"], "intake")[0]["payload"]
    console.print(f"ticket 2 (new thread) intake loaded: {intake}")
    render_result(second["result"])
    console.print(cases.customer_cases(cust, limit=5)[0]["subject"], "<- remembered history")


def growth(first: dict | None = None) -> None:
    console.print(Rule("4. Newly resolved case becomes searchable"))
    first = first or normal()
    hits = hybrid_search("VPN keeps disconnecting every few minutes shared drive", k=8)
    console.print([(h.ticket_id, h.source, h.similarity) for h in hits])
    live = [h for h in hits if h.source == "live"]
    ok = "OK" if live else "NOT FOUND"
    console.print(f"[bold]{ok}[/]: live case(s) in results: {[h.ticket_id for h in live]}")
    env = service.start_ticket(
        f"umbrella-{RUN}",
        "VPN disconnecting constantly",
        "VPN connection drops every few minutes, cannot reach the shared drive.",
        auto_accept=True,
    )
    env = drive(env, "All users, started this week.")
    used = [e["ticket_id"] for e in env["result"]["evidence"] if e["source"] == "live"]
    console.print(f"new ticket cited live (agent-resolved) cases as evidence: {used}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", choices=["normal", "clarify", "memory", "growth", "all"], default="all"
    )
    a = ap.parse_args()
    from support_agent.preflight import PreflightError, run_all

    try:
        run_all()
    except PreflightError as e:
        raise SystemExit(f"Setup problem: {e}") from None
    if a.scenario in ("normal", "all"):
        first = normal()
    if a.scenario in ("clarify", "all"):
        clarify()
    if a.scenario in ("memory", "all"):
        memory()
    if a.scenario in ("growth", "all"):
        growth(first if a.scenario == "all" else None)
