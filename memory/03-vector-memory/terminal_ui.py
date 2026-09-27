"""Rich terminal UI for the semantic-memory project."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import Settings
from models import AddMemoryResult, MemoryStats, SearchResult, VectorMemory

console = Console()


def show_start(settings: Settings) -> None:
    console.print(
        Panel(
            "[bold]Persistent memory retrieved by meaning, not an exact key.[/bold]\n\n"
            "[cyan]/remember[/cyan] explicitly embeds and stores a memory.\n"
            "[cyan]/search[/cyan] retrieves nearby meanings from local ChromaDB.\n"
            "[magenta]/chat[/magenta] retrieves first, then calls OpenAI with untrusted memory data.",
            title="[bold cyan]Memory #3 · Vector / Semantic Memory[/bold cyan]",
            border_style="cyan",
        )
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_row("Chroma path", str(settings.chroma_path))
    table.add_row("Collection", settings.collection_name)
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("Chat model", settings.model)
    table.add_row("Default top-k", str(settings.top_k))
    table.add_row("Chat minimum score", f"{settings.min_similarity:.2f}")
    console.print(Panel(table, title="Configuration", border_style="blue"))
    show_help()


def show_help() -> None:
    table = Table(title="What to type", header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Purpose")
    table.add_row("/remember <text>", "Store one explicit memory")
    table.add_row(
        "/remember --type <type> --source <source> --importance <1-5> <text>",
        "Store with attributes",
    )
    table.add_row("/search <query>", "Semantic search")
    table.add_row(
        "/search --top-k <n> --type <type> --min-score <score> <query>",
        "Filtered semantic search",
    )
    table.add_row("/memory <id>", "Get one memory by stable ID")
    table.add_row("/delete <id>", "Explicitly delete one memory")
    table.add_row("/list [type]", "List stored memories, optionally by type")
    table.add_row("/stats", "Show local collection statistics")
    table.add_row("/chat <message>", "Retrieve relevant memories, then call OpenAI")
    table.add_row("/help", "Show this table")
    table.add_row("/quit  or  exit", "Exit")
    console.print(table)
    console.print(
        Panel(
            "[bold]Try this first[/bold]\n"
            "[cyan]/remember --type incident checkout latency was caused by fraud-service CPU saturation[/cyan]\n"
            "[cyan]/remember --type preference use Python 3.12 for the memory series[/cyan]\n"
            "[cyan]/search what caused the checkout slowdown?[/cyan]",
            border_style="green",
        )
    )


def read_input() -> str:
    return console.input("\n[bold cyan]vector>[/bold cyan] ").strip()


def _short(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def show_memory(memory: VectorMemory | None, *, title: str = "Vector memory") -> None:
    if memory is None:
        console.print(
            Panel("[yellow]NOT FOUND[/yellow]", title=title, border_style="yellow")
        )
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("ID", memory.id)
    table.add_row("Type", memory.memory_type.value)
    table.add_row("Source", memory.source)
    table.add_row("Importance", str(memory.importance))
    table.add_row("Content", memory.content)
    table.add_row(
        "Metadata", json.dumps(memory.metadata, ensure_ascii=False, sort_keys=True)
    )
    table.add_row("Created", memory.created_at.isoformat())
    table.add_row("Updated", memory.updated_at.isoformat())
    console.print(Panel(table, title=title, border_style="green"))


def show_added(result: AddMemoryResult) -> None:
    title = "Memory stored" if result.created else "Duplicate reused"
    show_memory(result.memory, title=title)
    if not result.created:
        console.print(
            "[yellow]Normalized content already existed; no second vector was stored.[/yellow]"
        )


def show_search(results: list[SearchResult], query: str) -> None:
    table = Table(title=f"Semantic matches · {query}", header_style="bold magenta")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Source")
    table.add_column("Memory", overflow="fold")
    if not results:
        table.add_row("—", "—", "—", "—", "No matching memories")
    for result in results:
        table.add_row(
            f"{result.similarity_score:.3f}",
            f"{result.distance:.3f}",
            result.memory.memory_type.value,
            result.memory.source,
            _short(result.memory.content),
        )
    console.print(table)
    console.print(
        "[dim]Score is derived from cosine distance for ranking; "
        "it is not a probability or confidence.[/dim]"
    )


def show_memories(memories: list[VectorMemory]) -> None:
    table = Table(title="Stored vector memories", header_style="bold magenta")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Type", style="cyan")
    table.add_column("Importance", justify="right")
    table.add_column("Source")
    table.add_column("Content", overflow="fold")
    if not memories:
        table.add_row("—", "—", "—", "—", "No memories stored")
    for memory in memories:
        table.add_row(
            memory.id[:8],
            memory.memory_type.value,
            str(memory.importance),
            memory.source,
            _short(memory.content),
        )
    console.print(table)


def show_stats(stats: MemoryStats, path: Path) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="bold white")
    table.add_row("Memories", str(stats.count))
    table.add_row("Collection", stats.collection)
    table.add_row("Persistent path", str(path))
    console.print(Panel(table, title="Vector memory stats", border_style="blue"))


def show_chat(answer: str, memories: list[SearchResult]) -> None:
    table = Table(
        title=f"Relevant context used · {len(memories)} result(s)",
        header_style="bold magenta",
    )
    table.add_column("Score", style="green", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Source")
    table.add_column("Memory", overflow="fold")
    if not memories:
        table.add_row("—", "—", "—", "No memory passed the minimum score")
    for result in memories:
        table.add_row(
            f"{result.similarity_score:.3f}",
            result.memory.memory_type.value,
            result.memory.source,
            _short(result.memory.content),
        )
    console.print(table)
    console.print(
        "[dim]Only relevant context meeting the configured minimum score was used.[/dim]"
    )
    console.print(
        Panel(
            Markdown(answer),
            title="Answer",
            border_style="blue",
        )
    )


def show_deleted(deleted: bool) -> None:
    color = "green" if deleted else "yellow"
    text = "DELETED" if deleted else "NOT FOUND"
    console.print(
        Panel(f"[{color}]{text}[/{color}]", title="Delete", border_style=color)
    )


def show_error(message: str) -> None:
    console.print(Panel(message, title="Error", border_style="red"))


def show_usage(command: str, usage: str, example: str) -> None:
    console.print(
        Panel(
            f"[bold]Usage[/bold]\n{usage}\n\n[bold]Example[/bold]\n[cyan]{example}[/cyan]",
            title=f"Invalid {command} command",
            border_style="red",
        )
    )


def show_goodbye() -> None:
    console.print("[dim]Persistent vector memory remains on disk. Goodbye.[/dim]")
