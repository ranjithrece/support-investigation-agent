"""Command line interface."""

import typer
from rich.console import Console

from support_agent.config import get_settings

app = typer.Typer(add_completion=False, help="Autonomous Support Investigation Agent")
console = Console()

_BASE = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/"
_FILES = [
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "dataset-tickets-multi-lang-4-20k.csv",
]


@app.command()
def download() -> None:
    """Download the public Hugging Face dataset CSVs (no token needed)."""
    import urllib.request

    raw = get_settings().data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        dest = raw / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            console.print(f"[green]ok[/] {name} (cached)")
            continue
        console.print(f"downloading {name} ...")
        urllib.request.urlretrieve(_BASE + name, dest)
    console.print("[bold green]dataset ready[/]")


@app.command()
def ingest(
    limit: int | None = typer.Option(None, help="Index only N tickets (quick trial)"),
    holdout: int = typer.Option(200, help="Tickets reserved for evaluation (never indexed)"),
    rebuild: bool = typer.Option(False, help="Delete DB + vector index first"),
    skip_vectors: bool = typer.Option(False, help="SQLite/FTS only (no embeddings)"),
) -> None:
    """Ingest + index historical tickets (SQLite, FTS5, Chroma)."""
    from support_agent.ingest.pipeline import ingest as run
    from support_agent.preflight import PreflightError, check_ollama

    if not skip_vectors:
        try:
            check_ollama(need_chat=False)
        except PreflightError as e:
            console.print(f"[bold red]Setup problem:[/] {e}")
            raise typer.Exit(1) from None
    stats = run(limit=limit, holdout=holdout, rebuild=rebuild, skip_vectors=skip_vectors)
    console.print(stats)


def _preflight(need_chat: bool = True) -> None:
    from support_agent.preflight import PreflightError, run_all

    try:
        run_all(need_chat)
    except PreflightError as e:
        console.print(f"[bold red]Setup problem:[/] {e}")
        raise typer.Exit(1) from None


def _show_env(env: dict) -> None:
    from support_agent.render import render_question, render_result

    if env["status"] == "needs_input":
        render_question(env)
    else:
        render_result(env["result"])
        console.print(
            f"accepted={env.get('accepted')} indexed={env.get('indexed')} "
            f"memories_stored={len(env.get('stored_memories', []))}"
        )


@app.command()
def ticket(
    subject: str = typer.Option(..., help="Ticket subject"),
    body: str = typer.Option(..., help="Ticket body"),
    customer_id: str = typer.Option(..., help="Customer identifier (long-term memory key)"),
    thread_id: str | None = typer.Option(None, help="Reuse/choose a thread id"),
    auto_accept: bool = typer.Option(False, help="Skip the human accept/reject step"),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON envelope"),
) -> None:
    """Submit a new ticket. May pause to ask a clarification question."""
    import json

    from support_agent.service import start_ticket

    _preflight()
    env = start_ticket(customer_id, subject, body, thread_id, auto_accept)
    console.print_json(json.dumps(env, default=str)) if as_json else _show_env(env)


@app.command()
def chat(
    customer_id: str = typer.Option(..., help="Customer identifier (long-term memory key)"),
) -> None:
    """Interactive multi-turn session: open tickets, answer questions and accept in one prompt."""
    from support_agent.chat import chat_loop

    _preflight()
    chat_loop(customer_id)


@app.command()
def resume(
    thread_id: str = typer.Option(..., help="Thread id printed by `agent ticket`"),
    answer: str = typer.Option(..., help="Clarification answer, `accept`, or rejection feedback"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answer a pending clarification question or accept/reject a proposed outcome."""
    import json

    from support_agent.service import resume as run

    _preflight()
    env = run(thread_id, answer)
    console.print_json(json.dumps(env, default=str)) if as_json else _show_env(env)


@app.command()
def show(
    case_id: str | None = typer.Option(None),
    customer_id: str | None = typer.Option(None),
) -> None:
    """Show a case (with its event trail) or a customer's cases and memory."""
    import json

    from support_agent.memory.long_term import load_memories
    from support_agent.service import show_case
    from support_agent.storage import cases

    if case_id:
        console.print_json(json.dumps(show_case(case_id), default=str))
    elif customer_id:
        rows = cases.customer_cases(customer_id, limit=20)
        console.print_json(
            json.dumps(
                {"cases": rows, "memory": load_memories(customer_id, "", limit=30)}, default=str
            )
        )
    else:
        raise typer.BadParameter("give --case-id or --customer-id")


@app.command(name="eval")
def eval_cmd(
    limit: int = typer.Option(10, help="Number of hold-out tickets (+4 behaviour cases)"),
    out: str = typer.Option("evals/results/latest.json", help="Where to write the JSON report"),
) -> None:
    """Run LangSmith evaluators locally (no account / upload) on hold-out + behaviour cases."""
    import json
    from pathlib import Path

    from rich.table import Table

    from support_agent.evals.runner import run_evaluation

    _preflight()
    rows = run_evaluation(limit)
    keys = ["retrieval_relevance", "groundedness", "response_quality", "tool_usage", "behaviour"]
    t = Table(title="Evaluation")
    t.add_column("ticket")
    for k in keys:
        t.add_column(k)
    means: dict[str, list[float]] = {k: [] for k in keys}
    for r in rows:
        cells = []
        for k in keys:
            v = r["scores"].get(k)
            cells.append("-" if v is None else f"{v:.2f}")
            if v is not None:
                means[k].append(v)
        t.add_row(r["subject"], *cells)
    t.add_row("[bold]MEAN[/]", *[f"{sum(v) / len(v):.2f}" if v else "-" for v in means.values()])
    console.print(t)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2, default=str))
    console.print(f"report written to {out}")


if __name__ == "__main__":
    app()
