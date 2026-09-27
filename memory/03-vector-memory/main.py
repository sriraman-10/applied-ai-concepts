"""Interactive CLI for persistent vector / semantic memory."""

from __future__ import annotations

import asyncio
import logging
import shlex
from dataclasses import dataclass

from dotenv import load_dotenv

from agent import AgentRunError, EngineeringMemoryAgent
from config import Settings
from embeddings import EmbeddingError, OpenAIEmbeddingProvider
from memory import MemoryValidationError, VectorMemoryManager
from models import MemoryType
from terminal_ui import (
    console,
    read_input,
    show_added,
    show_chat,
    show_deleted,
    show_error,
    show_goodbye,
    show_help,
    show_memories,
    show_memory,
    show_search,
    show_start,
    show_stats,
    show_usage,
)

EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}


@dataclass(slots=True)
class RememberArgs:
    text: str
    memory_type: str = "observation"
    source: str = "cli"
    importance: int = 3


@dataclass(slots=True)
class SearchArgs:
    query: str
    top_k: int | None = None
    memory_type: str | None = None
    min_score: float | None = None


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError as error:
        raise MemoryValidationError(f"Could not parse command: {error}") from error


def _remember_args(parts: list[str]) -> RememberArgs:
    result = RememberArgs(text="")
    words: list[str] = []
    index = 1
    while index < len(parts):
        token = parts[index]
        if token in {"--type", "--source", "--importance"}:
            if index + 1 >= len(parts):
                raise MemoryValidationError(f"{token} requires a value")
            value = parts[index + 1]
            if token == "--type":
                result.memory_type = value
            elif token == "--source":
                result.source = value
            else:
                try:
                    result.importance = int(value)
                except ValueError as error:
                    raise MemoryValidationError(
                        "--importance must be an integer"
                    ) from error
            index += 2
        else:
            words.append(token)
            index += 1
    result.text = " ".join(words).strip()
    if not result.text:
        raise MemoryValidationError("/remember requires memory text")
    return result


def _search_args(parts: list[str]) -> SearchArgs:
    result = SearchArgs(query="")
    words: list[str] = []
    index = 1
    while index < len(parts):
        token = parts[index]
        if token in {"--top-k", "--type", "--min-score"}:
            if index + 1 >= len(parts):
                raise MemoryValidationError(f"{token} requires a value")
            value = parts[index + 1]
            try:
                if token == "--top-k":
                    result.top_k = int(value)
                elif token == "--min-score":
                    result.min_score = float(value)
                else:
                    result.memory_type = value
            except ValueError as error:
                raise MemoryValidationError(f"{token} has an invalid value") from error
            index += 2
        else:
            words.append(token)
            index += 1
    result.query = " ".join(words).strip()
    if not result.query:
        raise MemoryValidationError("/search requires a query")
    return result


async def run_cli() -> None:
    load_dotenv()
    try:
        settings = Settings.from_env()
        provider = OpenAIEmbeddingProvider(settings.embedding_model)
        memory = VectorMemoryManager(
            settings.chroma_path,
            settings.collection_name,
            provider,
            max_top_k=settings.max_top_k,
        )
    except (ValueError, OSError) as error:
        show_error(f"Could not initialize vector memory: {error}")
        return
    agent = EngineeringMemoryAgent(memory, settings)
    show_start(settings)
    while True:
        try:
            command = await asyncio.to_thread(read_input)
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            break
        if command == "/help":
            show_help()
            continue
        try:
            parts = _tokens(command)
            operation = parts[0]
            if operation == "/remember":
                args = _remember_args(parts)
                with console.status(
                    "[cyan]Embedding and storing explicit memory…[/cyan]"
                ):
                    result = await memory.add_memory(
                        args.text,
                        memory_type=args.memory_type,
                        source=args.source,
                        importance=args.importance,
                    )
                show_added(result)
            elif operation == "/search":
                args = _search_args(parts)
                filters = (
                    {"memory_type": args.memory_type} if args.memory_type else None
                )
                with console.status(
                    "[cyan]Embedding query and searching ChromaDB…[/cyan]"
                ):
                    results = await memory.search(
                        args.query,
                        top_k=args.top_k or settings.top_k,
                        filters=filters,
                        min_similarity=args.min_score,
                    )
                show_search(results, args.query)
            elif operation == "/memory":
                if len(parts) != 2:
                    show_usage("/memory", "/memory <id>", "/memory 7bfb…")
                    continue
                show_memory(memory.get_memory(parts[1]))
            elif operation == "/delete":
                if len(parts) != 2:
                    show_usage("/delete", "/delete <id>", "/delete 7bfb…")
                    continue
                show_deleted(memory.delete_memory(parts[1]))
            elif operation == "/list":
                if len(parts) > 2:
                    show_usage("/list", "/list [type]", "/list incident")
                    continue
                show_memories(
                    memory.list_memories(
                        memory_type=parts[1] if len(parts) == 2 else None
                    )
                )
            elif operation == "/stats":
                if len(parts) != 1:
                    show_usage("/stats", "/stats", "/stats")
                    continue
                show_stats(memory.stats(), settings.chroma_path)
            elif operation == "/chat":
                message = command.split(maxsplit=1)[1].strip() if len(parts) > 1 else ""
                if not message:
                    show_usage(
                        "/chat",
                        "/chat <message>",
                        "/chat What caused the last checkout slowdown?",
                    )
                    continue
                with console.status(
                    "[cyan]Retrieving relevant memory, then calling OpenAI…[/cyan]"
                ):
                    result = await agent.chat(message)
                show_chat(result.answer, result.memories)
            elif operation.startswith("/"):
                show_error("Unknown command. Type /help to see available commands.")
            else:
                show_error(
                    "Start with a command such as /remember, /search, or /chat. Type /help for examples."
                )
        except (
            MemoryValidationError,
            EmbeddingError,
            AgentRunError,
            ValueError,
        ) as error:
            show_error(str(error))
    show_goodbye()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
