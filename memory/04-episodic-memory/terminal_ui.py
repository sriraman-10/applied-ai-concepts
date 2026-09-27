"""Rich terminal presentation for episodic memory."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import Settings
from models import Episode, EpisodeSearchResult, EpisodeStats

console = Console()


def show_start(settings: Settings) -> None:
    console.print(
        Panel(
            "[bold]Complete support experiences stored in Postgres and retrieved with pgvector.[/bold]\n\n"
            "[cyan]Manual lifecycle[/cyan] records actions and observations explicitly.\n"
            "[magenta]/chat[/magenta] runs the real support-agent tool loop and stores only terminal cases.",
            title="[bold cyan]Memory #4 · Episodic Memory[/bold cyan]",
            border_style="cyan",
        )
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_row("Database", "PostgreSQL + pgvector")
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("Embedding dimensions", str(settings.embedding_dimensions))
    table.add_row("Chat model", settings.model)
    table.add_row(
        "Top-k / minimum score", f"{settings.top_k} / {settings.min_similarity:.2f}"
    )
    console.print(Panel(table, title="Configuration", border_style="blue"))
    show_help()


def show_help() -> None:
    table = Table(title="What to type", header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Purpose")
    table.add_row("/start <issue>", "Start a manual support episode")
    table.add_row(
        "/action <action> | <result>", "Record an observable action and result"
    )
    table.add_row("/observe <source> | <observation>", "Record an observation")
    table.add_row("/resolve <resolution>", "Complete the current episode as resolved")
    table.add_row("/escalate <reason>", "Complete the current episode as escalated")
    table.add_row("/search <query>", "Find similar resolved episodes with pgvector")
    table.add_row("/episode <case-id>", "Retrieve one complete Postgres row")
    table.add_row("/list", "List episodes")
    table.add_row("/stats", "Show lifecycle counts")
    table.add_row("/chat <issue>", "Run with episodic retrieval and persist the case")
    table.add_row(
        "/chat --no-memory <issue>", "A/B baseline: run without episodic retrieval"
    )
    table.add_row("/help", "Show this table")
    table.add_row("/quit  or  exit", "Exit")
    console.print(table)
    console.print(
        Panel(
            "[bold]Agent demo[/bold]\n"
            "[cyan]/chat Payment failed but money was deducted for TXN-DEMO-001[/cyan]\n"
            "Run a similar case later to see the completed episode retrieved.",
            title="Try this first",
            border_style="green",
        )
    )


def read_input() -> str:
    return console.input("\n[bold cyan]episode>[/bold cyan] ").strip()


def _short(value: str, limit: int = 90) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def show_episode(episode: Episode | None, *, title: str = "Support episode") -> None:
    if episode is None:
        console.print(
            Panel("[yellow]NOT FOUND[/yellow]", title=title, border_style="yellow")
        )
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("Case", episode.episode_key)
    table.add_row("Status", episode.status.value)
    table.add_row("Outcome", episode.outcome.value if episode.outcome else "—")
    table.add_row("Issue type", episode.issue_type)
    table.add_row("Description", episode.description)
    table.add_row("Actions", str(len(episode.actions)))
    table.add_row("Observations", str(len(episode.observations)))
    table.add_row("Resolution", episode.resolution or "—")
    table.add_row("Summary", episode.episode_summary or "Created only after completion")
    table.add_row("Created", episode.created_at.isoformat())
    table.add_row(
        "Completed", episode.completed_at.isoformat() if episode.completed_at else "—"
    )
    console.print(Panel(table, title=title, border_style="green"))
    if episode.actions:
        actions = Table(title="Ordered actions", header_style="bold magenta")
        actions.add_column("#", justify="right")
        actions.add_column("Action", style="cyan")
        actions.add_column("Tool")
        actions.add_column("Result", overflow="fold")
        for item in episode.actions:
            actions.add_row(
                str(item.sequence),
                item.action_type,
                item.tool_name or "—",
                item.result_summary or "—",
            )
        console.print(actions)
    if episode.observations:
        observations = Table(title="Observations", header_style="bold magenta")
        observations.add_column("#", justify="right")
        observations.add_column("Source", style="cyan")
        observations.add_column("Important")
        observations.add_column("Content", overflow="fold")
        for item in episode.observations:
            observations.add_row(
                str(item.sequence),
                item.source,
                "yes" if item.important else "no",
                item.content,
            )
        console.print(observations)


def show_search(
    results: list[EpisodeSearchResult], title: str = "Similar completed episodes"
) -> None:
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Case", style="cyan")
    table.add_column("Issue type")
    table.add_column("Outcome")
    table.add_column("Issue / resolution", overflow="fold")
    if not results:
        table.add_row("—", "—", "—", "—", "No resolved episode met the threshold")
    for result in results:
        episode = result.episode
        table.add_row(
            f"{result.similarity_score:.3f}",
            episode.episode_key,
            episode.issue_type,
            episode.outcome.value if episode.outcome else "—",
            _short(f"{episode.description} → {episode.resolution}"),
        )
    console.print(table)
    console.print(
        "[dim]Cosine-derived scores rank episodes; they are not probabilities.[/dim]"
    )


def show_list(episodes: list[Episode]) -> None:
    table = Table(title="Support episodes", header_style="bold magenta")
    table.add_column("Case", style="cyan")
    table.add_column("Status")
    table.add_column("Outcome")
    table.add_column("Type")
    table.add_column("Issue", overflow="fold")
    if not episodes:
        table.add_row("—", "—", "—", "—", "No episodes")
    for episode in episodes:
        table.add_row(
            episode.episode_key,
            episode.status.value,
            episode.outcome.value if episode.outcome else "—",
            episode.issue_type,
            _short(episode.description),
        )
    console.print(table)


def show_stats(stats: EpisodeStats) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")
    for name in (
        "total",
        "open",
        "completed",
        "resolved",
        "escalated",
        "failed",
        "abandoned",
    ):
        table.add_row(name.title(), str(getattr(stats, name)))
    console.print(Panel(table, title="Episodic memory stats", border_style="blue"))


def show_tool_event(
    name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> None:
    table = Table(show_header=True, box=None, header_style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Arguments", overflow="fold")
    table.add_column("Observation", overflow="fold")
    table.add_row(
        name,
        json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        json.dumps(result, ensure_ascii=False, sort_keys=True),
    )
    console.print(table)


def show_chat(
    answer: str,
    episode: Episode,
    similar: list[EpisodeSearchResult],
    *,
    memory_enabled: bool,
) -> None:
    if memory_enabled:
        show_search(
            similar, title=f"Past experience supplied · {len(similar)} result(s)"
        )
    else:
        console.print(
            Panel(
                "Episodic retrieval was disabled for this A/B baseline run.",
                title="No-memory baseline",
                border_style="yellow",
            )
        )
    console.print(Panel(Markdown(answer), title="Support answer", border_style="blue"))
    console.print(
        Panel(
            f"Stored [bold]{episode.episode_key}[/bold] · outcome: [bold]{episode.outcome.value}[/bold]",
            title="Episode completed",
            border_style="green",
        )
    )


def show_error(message: str) -> None:
    console.print(Panel(message, title="Error", border_style="red"))


def show_info(message: str, title: str) -> None:
    console.print(Panel(message, title=title, border_style="green"))


def show_goodbye() -> None:
    console.print("[dim]PostgreSQL connection pool closed. Goodbye.[/dim]")
