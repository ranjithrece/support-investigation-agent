"""Rich rendering of agent results for the CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def render_result(result: dict[str, Any]) -> None:
    c = result["classification"]
    t = Table.grid(padding=(0, 2))
    t.add_row("[bold]Case[/]", result["case_id"])
    t.add_row(
        "[bold]Classification[/]", f"{c['queue']} / {c['type']} / {c['priority']}  tags={c['tags']}"
    )
    t.add_row("[bold]Confidence[/]", f"{result['confidence']:.2f}")
    esc = result["escalation"]
    t.add_row(
        "[bold]Escalation[/]",
        f"{'YES - ' + str(esc.get('reason')) if esc['required'] else 'no'}"
        + (f" -> {esc.get('target_queue')}" if esc["required"] else ""),
    )
    v = result.get("verification") or {}
    t.add_row("[bold]Grounded[/]", f"{v.get('grounded')} (score {v.get('grounding_score')})")
    console.print(Panel(t, title="Result", border_style="green"))
    console.print(Panel(result["analysis"], title="Analysis"))
    console.print(Panel(result["resolution"], title="Resolution"))
    ev = Table(title="Evidence")
    for col in ("ticket", "src", "queue/type", "sim", "summary"):
        ev.add_column(col)
    for e in result["evidence"]:
        ev.add_row(
            e["ticket_id"],
            e["source"],
            f"{e['queue']}/{e['type']}",
            f"{e['similarity']:.2f}",
            e["summary"][:70],
        )
    console.print(ev)


def render_question(env: dict[str, Any], hint: bool = True) -> None:
    i = env["interrupt"]
    if i["type"] == "clarification":
        console.print(
            Panel(
                i["question"],
                title=f"Agent needs more information ({env['case_id']})",
                border_style="yellow",
            )
        )
        if hint:
            console.print(
                f'Reply with:  agent resume --thread-id {env["thread_id"]} --answer "..."'
            )
    else:
        render_result(i["result"])
        console.print(Panel("Accept this outcome?", border_style="yellow"))
        if hint:
            console.print(
                f"Reply with:  agent resume --thread-id {env['thread_id']} --answer accept"
                "   (or type feedback to reject)"
            )
