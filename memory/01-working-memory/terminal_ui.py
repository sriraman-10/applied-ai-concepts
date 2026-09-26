"""Rich terminal presentation for the working-memory demo."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import Settings
from memory import CompactionReport, MemoryItem, MemoryStats, Priority

console = Console()


def show_start(settings: Settings) -> None:
    console.print(
        Panel(
            "[bold]An interactive agent with application-managed working memory.[/bold]\n\n"
            "[cyan]Type a normal sentence[/cyan] to call OpenAI and add a conversation turn.\n"
            "[magenta]Type a /command[/magenta] to inspect or modify memory without calling OpenAI.",
            title="[bold cyan]Memory #1 · Working / In-Context Memory[/bold cyan]",
            border_style="cyan",
        )
    )
    settings_table = Table(show_header=False, box=None, padding=(0, 2))
    settings_table.add_column(style="dim")
    settings_table.add_column(style="bold white")
    settings_table.add_row("Model", settings.model)
    settings_table.add_row("Hard limit", f"{settings.hard_token_limit} tokens")
    settings_table.add_row("Compaction target", f"{settings.target_token_limit} tokens")
    settings_table.add_row("Protected recent items", str(settings.keep_recent_items))
    console.print(Panel(settings_table, title="Configuration", border_style="blue"))
    show_help()


def show_help() -> None:
    table = Table(title="What to type", header_style="bold magenta")
    table.add_column("Input", style="cyan", no_wrap=True)
    table.add_column("What happens")
    table.add_row("Any normal sentence", "Calls OpenAI, then stores the user and assistant messages")
    table.add_row("/stats", "Shows token usage, item count, summary size, and compactions")
    table.add_row("/memory", "Shows every item currently kept in working memory")
    table.add_row("/summary", "Shows the compacted summary after compaction occurs")
    table.add_row(
        "/tool important <text>",
        "Adds important simulated tool evidence; protected from noise eviction",
    )
    table.add_row(
        "/tool noise <text>",
        "Adds low-value simulated output; first candidate for eviction",
    )
    table.add_row("/help", "Shows this table again")
    table.add_row("/quit  or  exit", "Ends the program")
    console.print(table)
    console.print(
        Panel(
            "[bold]Try this first:[/bold]\n"
            "[cyan]I am investigating checkout latency. Remember that payment-service is involved.[/cyan]\n\n"
            "Then inspect it with [magenta]/memory[/magenta] and [magenta]/stats[/magenta].",
            title="Example",
            border_style="green",
        )
    )


def read_input() -> str:
    return console.input("\n[bold cyan]you>[/bold cyan] ").strip()


def show_assistant(answer: str) -> None:
    console.print(Panel(Markdown(answer), title="Assistant", border_style="blue"))


def show_stats(stats: MemoryStats, settings: Settings) -> None:
    percentage = min(100, round(100 * stats.estimated_tokens / settings.hard_token_limit))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")
    table.add_row("Estimated tokens", f"{stats.estimated_tokens} / {settings.hard_token_limit} ({percentage}%)")
    table.add_row("Target after compaction", str(settings.target_token_limit))
    table.add_row("Live items", str(stats.item_count))
    table.add_row("Summary tokens", str(stats.summary_tokens))
    table.add_row("Summary exists", "yes" if stats.has_summary else "no")
    table.add_row("Compactions", str(stats.compaction_count))
    color = "yellow" if percentage >= 75 else "green"
    console.print(Panel(table, title="Working-memory stats", border_style=color))


def _priority_style(priority: Priority) -> str:
    if priority is Priority.HIGH:
        return "bold green"
    if priority is Priority.LOW:
        return "dim yellow"
    return "white"


def show_memory(items: tuple[MemoryItem, ...]) -> None:
    if not items:
        console.print(
            Panel(
                "No items yet. Type a normal sentence to execute the agent and create the first turn.",
                title="Working memory",
                border_style="yellow",
            )
        )
        return
    table = Table(title="Current working-memory items", header_style="bold magenta")
    table.add_column("ID", style="dim cyan", no_wrap=True)
    table.add_column("Time (UTC)", style="dim", no_wrap=True)
    table.add_column("Kind", style="cyan", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Content", overflow="fold")
    for item in items:
        priority = Text(item.priority.name, style=_priority_style(item.priority))
        table.add_row(
            item.id[:8],
            item.timestamp.strftime("%H:%M:%S"),
            item.kind.value,
            priority,
            str(item.estimated_tokens),
            item.content,
        )
    console.print(table)


def show_summary(summary: str | None) -> None:
    if summary:
        console.print(Panel(Markdown(summary), title="Compacted summary", border_style="magenta"))
        return
    console.print(
        Panel(
            "No summary exists yet. Continue the conversation until the hard token limit is reached.",
            title="Compacted summary",
            border_style="yellow",
        )
    )


def show_compaction(report: CompactionReport) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")
    table.add_row("Before", f"{report.before_tokens} tokens")
    table.add_row("After", f"{report.after_tokens} tokens")
    table.add_row("Noise / hard-evicted items", str(report.items_dropped))
    table.add_row("Items merged into summary", str(report.items_summarized))
    table.add_row("Compaction number", str(report.compaction_count))
    if report.summary_error:
        table.add_row("Summary error", report.summary_error)
    console.print(Panel(table, title="Memory compaction", border_style="magenta"))


def show_tool_added(*, important: bool, content: str) -> None:
    label = "IMPORTANT" if important else "LOW-VALUE NOISE"
    color = "green" if important else "yellow"
    console.print(
        Panel(
            content,
            title=f"Tool observation added · [{color}]{label}[/{color}]",
            border_style=color,
        )
    )


def show_usage_error() -> None:
    console.print(
        Panel(
            "Include an observation after the priority.\n\n"
            "[bold]Examples[/bold]\n"
            "[cyan]/tool important checkout-service p95 latency is 1850ms[/cyan]\n"
            "[cyan]/tool noise health-check endpoint returned 200[/cyan]",
            title="Invalid /tool command",
            border_style="red",
        )
    )


def show_error(message: str) -> None:
    console.print(Panel(message, title="Error", border_style="red"))


def show_goodbye() -> None:
    console.print("[dim]Goodbye.[/dim]")
