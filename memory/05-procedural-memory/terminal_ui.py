"""Rich terminal presentation for procedural memory."""

from __future__ import annotations

import json
from typing import Any
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import Settings
from models import (
    ProcedureExecution,
    ProcedureSearchResult,
    ProcedureStats,
    SupportProcedure,
)

console = Console()


def show_start(settings: Settings) -> None:
    console.print(
        Panel(
            "[bold]Approved support playbooks retrieved from Postgres and executed as structured steps.[/bold]\n\n"
            "[cyan]Procedure definitions[/cyan] are persistent, versioned knowledge.\n"
            "[magenta]Executions[/magenta] are separate runtime records with hard application safety checks.",
            title="[bold cyan]Memory #5 · Procedural Memory[/bold cyan]",
            border_style="cyan",
        )
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    for key, value in (
        ("Database", "PostgreSQL + pgvector"),
        ("Embedding model", settings.embedding_model),
        ("Chat model", settings.model),
        ("Top-k / minimum score", f"{settings.top_k} / {settings.min_similarity:.2f}"),
    ):
        table.add_row(key, value)
    console.print(Panel(table, title="Configuration", border_style="blue"))
    show_help()


def show_help() -> None:
    table = Table(title="What to type", header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Purpose")
    for command, purpose in (
        ("/seed", "Create the three approved demo procedures"),
        ("/procedures", "List every procedure version"),
        ("/procedure <key> [version]", "Show a procedure; defaults to active version"),
        ("/search <issue>", "Rank relevant active procedures"),
        ("/activate <key> <version>", "Admin/demo: activate exactly one version"),
        (
            "/deactivate <key> <version>",
            "Admin/demo: return an active version to draft",
        ),
        ("/executions", "List procedure runs"),
        ("/execution <id>", "Show one run and its steps"),
        ("/chat <issue>", "Select and execute one approved procedure"),
        ("/stats", "Show lifecycle counts"),
        ("/help", "Show this table"),
        ("/quit  or  exit", "Exit"),
    ):
        table.add_row(command, purpose)
    console.print(table)
    console.print(
        Panel(
            "[cyan]/seed[/cyan]\n[cyan]/search Checkout failed but my card was charged[/cyan]\n[cyan]/chat Checkout failed but my card was charged for TXN-DEMO-001[/cyan]",
            title="Try this first",
            border_style="green",
        )
    )


def read_input() -> str:
    return console.input("\n[bold cyan]procedure>[/bold cyan] ").strip()


def show_procedures(items: list[SupportProcedure]) -> None:
    table = Table(title="Support procedures", header_style="bold magenta")
    for col in ("Key", "Version", "Status", "Category", "Name"):
        table.add_column(col)
    if not items:
        table.add_row("—", "—", "—", "—", "No procedures; run /seed")
    for p in items:
        table.add_row(
            p.procedure_key, str(p.version), p.status.value, p.category, p.name
        )
    console.print(table)


def show_procedure(p: SupportProcedure | None) -> None:
    if p is None:
        console.print(Panel("NOT FOUND", border_style="yellow"))
        return
    info = Table(show_header=False, box=None)
    info.add_column(style="cyan")
    info.add_column(overflow="fold")
    for k, v in (
        ("Key", p.procedure_key),
        ("Version", str(p.version)),
        ("Status", p.status.value),
        ("Category", p.category),
        ("Description", p.description),
        ("Summary", p.procedure_summary),
        ("Completion", "; ".join(p.completion_conditions)),
        ("Escalation", "; ".join(p.escalation_rules)),
    ):
        info.add_row(k, v)
    console.print(Panel(info, title=p.name, border_style="green"))
    steps = Table(title="Approved ordered steps")
    steps.add_column("#")
    steps.add_column("Action", style="cyan")
    steps.add_column("Tool")
    steps.add_column("Required")
    for s in p.steps:
        steps.add_row(
            str(s.sequence), s.action, s.tool or "—", "yes" if s.required else "no"
        )
    console.print(steps)


def show_search(
    results: list[ProcedureSearchResult], title: str = "Active procedure candidates"
) -> None:
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Key", style="cyan")
    table.add_column("Version")
    table.add_column("Category")
    table.add_column("Name")
    if not results:
        table.add_row("—", "—", "—", "—", "No active procedure met the threshold")
    for r in results:
        table.add_row(
            f"{r.similarity_score:.3f}",
            r.procedure.procedure_key,
            str(r.procedure.version),
            r.procedure.category,
            r.procedure.name,
        )
    console.print(table)
    console.print(
        "[dim]Cosine-derived scores rank procedures; they are not probabilities.[/dim]"
    )


def show_execution(e: ProcedureExecution | None) -> None:
    if e is None:
        console.print(Panel("NOT FOUND", border_style="yellow"))
        return
    info = Table(show_header=False, box=None)
    info.add_column(style="cyan")
    info.add_column()
    for k, v in (
        ("Execution", e.id),
        ("Procedure", f"{e.procedure_key} v{e.procedure_version}"),
        ("Status", e.status.value),
        ("Current step", str(e.current_step)),
        ("Outcome", e.outcome or "—"),
    ):
        info.add_row(k, v)
    console.print(Panel(info, title="Procedure execution", border_style="blue"))
    steps = Table(title="Steps executed")
    steps.add_column("#")
    steps.add_column("Action", style="cyan")
    steps.add_column("Tool")
    steps.add_column("Status")
    steps.add_column("Result", overflow="fold")
    for s in e.steps:
        steps.add_row(
            str(s.step_sequence),
            s.action,
            s.tool_name or "—",
            s.status.value,
            s.result_summary or "—",
        )
    console.print(steps)


def show_executions(items: list[ProcedureExecution]) -> None:
    table = Table(title="Procedure executions")
    table.add_column("ID", style="cyan")
    table.add_column("Procedure")
    table.add_column("Status")
    table.add_column("Outcome")
    if not items:
        table.add_row("—", "—", "—", "No executions")
    for e in items:
        table.add_row(
            e.id,
            f"{e.procedure_key} v{e.procedure_version}",
            e.status.value,
            e.outcome or "—",
        )
    console.print(table)


def show_tool_event(
    name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> None:
    table = Table(show_header=True, box=None)
    table.add_column("Tool", style="cyan")
    table.add_column("Arguments")
    table.add_column("Observation")
    table.add_row(
        name, json.dumps(arguments, sort_keys=True), json.dumps(result, sort_keys=True)
    )
    console.print(table)


def show_chat(result) -> None:
    show_search(result.candidates, "Ranked candidates")
    if result.selected:
        console.print(
            Panel(
                f"[bold]{result.selected.procedure.procedure_key}[/bold] · version {result.selected.procedure.version} · similarity {result.selected.similarity_score:.3f}",
                title="Selected procedure",
                border_style="green",
            )
        )
    console.print(
        Panel(Markdown(result.answer), title="Support answer", border_style="blue")
    )
    if result.execution:
        show_execution(result.execution)


def show_stats(s: ProcedureStats) -> None:
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column(justify="right", style="bold")
    for name in (
        "total",
        "draft",
        "active",
        "deprecated",
        "executions",
        "completed",
        "escalated",
    ):
        table.add_row(name.title(), str(getattr(s, name)))
    console.print(Panel(table, title="Procedural memory stats", border_style="blue"))


def show_info(message: str, title: str = "Done") -> None:
    console.print(Panel(message, title=title, border_style="green"))


def show_error(message: str) -> None:
    console.print(Panel(message, title="Error", border_style="red"))


def show_goodbye() -> None:
    console.print("[dim]PostgreSQL connection pool closed. Goodbye.[/dim]")
