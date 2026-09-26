"""Rich terminal UI for deterministic KV memory operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import Settings
from models import KVMemory, MemoryStats

console = Console()


def show_start(settings: Settings) -> None:
    console.print(
        Panel(
            "[bold]Persistent, deterministic memory addressed by structured keys.[/bold]\n\n"
            "[cyan]Local commands[/cyan] read and write SQLite without calling OpenAI.\n"
            "[magenta]/chat[/magenta] injects caller-selected keys before a model call.\n"
            "[magenta]/agent[/magenta] lets the model request validated KV tools.",
            title="[bold cyan]Memory #2 · External Key-Value Memory[/bold cyan]",
            border_style="cyan",
        )
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_row("Database", str(settings.database_path))
    table.add_row("Model", settings.model)
    table.add_row("Maximum TTL", f"{settings.max_ttl_seconds} seconds")
    console.print(Panel(table, title="Configuration", border_style="blue"))
    show_help()


def show_help() -> None:
    table = Table(title="What to type", header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Purpose")
    table.add_row("/set <ns> <entity> <key> <json> [ttl]", "Create or update an exact key")
    table.add_row("/get <ns> <entity> <key>", "Deterministic exact lookup")
    table.add_row("/delete <ns> <entity> <key>", "Explicitly delete one key")
    table.add_row("/list <ns> <entity>", "List active keys for one indexed scope")
    table.add_row("/clear <ns> [entity]", "Clear a namespace or one entity within it")
    table.add_row("/cleanup", "Physically remove expired rows")
    table.add_row("/stats", "Show active, expired, namespace, and entity counts")
    table.add_row("/chat <ns> <entity> <keys> <message>", "Application-managed retrieval")
    table.add_row("/agent <message>", "Tool-managed memory through OpenAI function calls")
    table.add_row("/help", "Show this table")
    table.add_row("/quit  or  exit", "Close SQLite and exit")
    console.print(table)
    console.print(
        Panel(
            "[bold]Persistence demo[/bold]\n"
            "[cyan]/set project memory-series python_version 3.12[/cyan]\n"
            "[cyan]/get project memory-series python_version[/cyan]\n"
            "Quit, restart, and run the same /get command.",
            title="Try this first",
            border_style="green",
        )
    )


def read_input() -> str:
    return console.input("\n[bold cyan]kv>[/bold cyan] ").strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def show_memory(memory: KVMemory | None, *, title: str = "KV memory") -> None:
    if memory is None:
        console.print(Panel("[yellow]NOT FOUND[/yellow]", title=title, border_style="yellow"))
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("Address", f"({memory.namespace}, {memory.entity_id}, {memory.key})")
    table.add_row("Value", _json(memory.value))
    table.add_row("Type", memory.value_type.value)
    table.add_row("Created", memory.created_at.isoformat())
    table.add_row("Updated", memory.updated_at.isoformat())
    table.add_row("Expires", memory.expires_at.isoformat() if memory.expires_at else "never")
    console.print(Panel(table, title=title, border_style="green"))


def show_memories(memories: list[KVMemory], namespace: str, entity_id: str) -> None:
    table = Table(
        title=f"KV memories · {namespace} / {entity_id}",
        header_style="bold magenta",
    )
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Value", overflow="fold")
    table.add_column("Expires", no_wrap=True)
    if not memories:
        table.add_row("—", "—", "No active memories", "—")
    else:
        for memory in memories:
            table.add_row(
                memory.key,
                memory.value_type.value,
                _json(memory.value),
                memory.expires_at.isoformat(timespec="seconds") if memory.expires_at else "never",
            )
    console.print(table)


def show_stats(stats: MemoryStats, database_path: Path) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")
    table.add_row("Active entries", str(stats.active_entries))
    table.add_row("Expired rows awaiting cleanup", str(stats.expired_entries))
    table.add_row("Namespaces", str(stats.namespaces))
    table.add_row("Namespace/entity pairs", str(stats.entities))
    table.add_row("Database", str(database_path))
    console.print(Panel(table, title="KV memory stats", border_style="blue"))


def show_count(message: str, count: int) -> None:
    console.print(Panel(f"{message}: [bold]{count}[/bold]", border_style="green"))


def show_deleted(deleted: bool) -> None:
    color = "green" if deleted else "yellow"
    text = "DELETED" if deleted else "NOT FOUND"
    console.print(Panel(f"[{color}]{text}[/{color}]", title="Delete", border_style=color))


def show_agent_answer(answer: str, mode: str) -> None:
    console.print(Panel(Markdown(answer), title=mode, border_style="blue"))


def show_tool_event(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Address / arguments")
    table.add_column("Result")
    address = "/".join(
        str(arguments.get(field, "")) for field in ("namespace", "entity_id", "key")
        if field in arguments
    )
    if name == "list_memories":
        summary = f"{result.get('count', 0)} active memories"
    elif name == "get_memory":
        summary = "found" if result.get("found") else "not found"
    elif name == "set_memory":
        summary = "stored" if result.get("stored") else str(result.get("error", "failed"))
    elif name == "delete_memory":
        summary = "deleted" if result.get("deleted") else "not found"
    else:
        summary = str(result.get("error", "completed"))
    table.add_row(name, address or _json(arguments), summary)
    console.print(table)


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
    console.print("[dim]SQLite connection closed. Goodbye.[/dim]")
