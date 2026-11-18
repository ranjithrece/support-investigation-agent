"""Interactive multi-turn session: one process, prompts instead of separate resume commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console

from support_agent import service
from support_agent.render import render_question, render_result

EXIT_WORDS = {"/exit", "/quit", "exit", "quit"}
HELP = (
    "Describe a problem to open a ticket. The agent may ask follow-up questions; answer them here.\n"
    "At the end type 'accept' to confirm, or type feedback to make it re-investigate.\n"
    "Commands: /new = abandon the current ticket, /exit = quit. Memory is kept per customer."
)


def _subject(text: str, limit: int = 80) -> str:
    first = text.strip().splitlines()[0] if text.strip() else "Support request"
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def chat_loop(
    customer_id: str,
    read: Callable[[str], str] = input,
    console: Console | None = None,
    app: Any = None,
) -> int:
    """Run the session; returns the number of tickets completed. `read` is injectable for tests."""
    out = console or Console()
    out.print(f"[bold]Support agent[/] - customer [cyan]{customer_id}[/]\n{HELP}\n")
    completed = 0
    env: dict[str, Any] | None = None

    while True:
        try:
            if env is None:
                text = read("\nyou (new ticket)> ").strip()
                if not text:
                    continue
                if text.lower() in EXIT_WORDS:
                    break
                out.print("[dim]investigating...[/]")
                env = service.start_ticket(customer_id, _subject(text), text, app=app)
                continue

            if env["status"] == "completed":
                render_result(env["result"])
                out.print(
                    f"accepted={env.get('accepted')} indexed={env.get('indexed')} "
                    f"memories_stored={len(env.get('stored_memories', []))}"
                )
                completed += 1
                env = None
                continue

            # status == needs_input
            render_question(env, hint=False)
            prompt = (
                "you (answer)> "
                if env["interrupt"]["type"] == "clarification"
                else "you (accept / feedback)> "
            )
            answer = read(prompt).strip()
            if answer.lower() in EXIT_WORDS:
                out.print(
                    f"Left ticket open. Resume later: "
                    f'agent resume --thread-id {env["thread_id"]} --answer "..."'
                )
                break
            if answer.lower() == "/new":
                out.print("[dim]ticket abandoned (kept as open case)[/]")
                env = None
                continue
            if not answer:
                continue
            out.print("[dim]thinking...[/]")
            env = service.resume(env["thread_id"], answer, app=app)
        except (EOFError, KeyboardInterrupt):
            out.print("\nbye")
            break
    return completed
