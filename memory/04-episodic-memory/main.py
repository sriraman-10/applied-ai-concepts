"""Interactive CLI for support-agent episodic memory."""

from __future__ import annotations

import asyncio
import logging
import shlex

from dotenv import load_dotenv

from agent import AgentRunError, SupportAgent, infer_issue_type
from config import Settings
from database import DatabaseError, PostgresDatabase
from embeddings import EmbeddingError, OpenAIEmbeddingProvider
from memory import EpisodeValidationError, EpisodicMemoryManager
from models import EpisodeOutcome
from store import PostgresEpisodeStore
from terminal_ui import (
    console,
    read_input,
    show_chat,
    show_episode,
    show_error,
    show_goodbye,
    show_help,
    show_info,
    show_list,
    show_search,
    show_start,
    show_stats,
    show_tool_event,
)
from tools import SupportTools

EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}


def _parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError as error:
        raise EpisodeValidationError(f"Could not parse command: {error}") from error


def _argument(command: str, operation: str) -> str:
    value = command[len(operation) :].strip()
    if not value:
        raise EpisodeValidationError(f"{operation} requires a value")
    return value


def _pipe_values(command: str, operation: str, count: int) -> list[str]:
    values = [value.strip() for value in _argument(command, operation).split("|")]
    if len(values) != count or any(not value for value in values):
        raise EpisodeValidationError(
            f"{operation} requires {count} values separated by |"
        )
    return values


async def run_cli() -> None:
    load_dotenv()
    database: PostgresDatabase | None = None
    try:
        settings = Settings.from_env()
        database = PostgresDatabase(
            settings.database_url,
            settings.embedding_dimensions,
            settings.migrations_path,
        )
        with console.status(
            "[cyan]Connecting to PostgreSQL and applying migrations…[/cyan]"
        ):
            await database.open()
        provider = OpenAIEmbeddingProvider(
            settings.embedding_model, settings.embedding_dimensions
        )
        memory = EpisodicMemoryManager(
            PostgresEpisodeStore(database), provider, max_top_k=settings.max_top_k
        )
    except (ValueError, OSError, DatabaseError) as error:
        show_error(str(error))
        return

    agent = SupportAgent(
        memory, settings, SupportTools(), on_tool_event=show_tool_event
    )
    current_episode_key: str | None = None
    show_start(settings)
    try:
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
                parts = _parts(command)
                operation = parts[0]
                if operation == "/start":
                    if current_episode_key is not None:
                        raise EpisodeValidationError(
                            f"finish current episode {current_episode_key} first"
                        )
                    issue = _argument(command, operation)
                    episode = await memory.start_episode(
                        issue,
                        issue_type=infer_issue_type(issue),
                        metadata={"channel": "manual_cli"},
                    )
                    current_episode_key = episode.episode_key
                    show_episode(episode, title="Episode started")
                elif operation == "/action":
                    if current_episode_key is None:
                        raise EpisodeValidationError("start an episode first")
                    action, result = _pipe_values(command, operation, 2)
                    episode = await memory.record_action(
                        current_episode_key,
                        action_type=action,
                        input_summary="manual CLI action",
                        result_summary=result,
                    )
                    show_info(
                        f"Recorded action #{len(episode.actions)}", current_episode_key
                    )
                elif operation == "/observe":
                    if current_episode_key is None:
                        raise EpisodeValidationError("start an episode first")
                    source, content = _pipe_values(command, operation, 2)
                    episode = await memory.record_observation(
                        current_episode_key, source=source, content=content
                    )
                    show_info(
                        f"Recorded observation #{len(episode.observations)}",
                        current_episode_key,
                    )
                elif operation in {"/resolve", "/escalate"}:
                    if current_episode_key is None:
                        raise EpisodeValidationError("start an episode first")
                    resolution = _argument(command, operation)
                    outcome = (
                        EpisodeOutcome.RESOLVED
                        if operation == "/resolve"
                        else EpisodeOutcome.ESCALATED
                    )
                    with console.status(
                        "[cyan]Building summary, embedding, and completing episode…[/cyan]"
                    ):
                        episode = await memory.complete_episode(
                            current_episode_key, resolution=resolution, outcome=outcome
                        )
                    current_episode_key = None
                    show_episode(episode, title="Episode completed")
                elif operation == "/search":
                    query = _argument(command, operation)
                    with console.status(
                        "[cyan]Embedding query and searching pgvector…[/cyan]"
                    ):
                        results = await memory.search_similar(
                            query,
                            top_k=settings.top_k,
                            min_similarity=settings.min_similarity,
                        )
                    show_search(results)
                elif operation == "/episode":
                    show_episode(
                        await memory.get_episode(_argument(command, operation))
                    )
                elif operation == "/list":
                    if len(parts) != 1:
                        raise EpisodeValidationError("usage: /list")
                    show_list(await memory.list_episodes())
                elif operation == "/stats":
                    if len(parts) != 1:
                        raise EpisodeValidationError("usage: /stats")
                    show_stats(await memory.stats())
                elif operation == "/chat":
                    chat_input = _argument(command, operation)
                    use_memory = True
                    if chat_input == "--no-memory":
                        raise EpisodeValidationError(
                            "/chat --no-memory requires a support issue"
                        )
                    if chat_input.startswith("--no-memory "):
                        use_memory = False
                        chat_input = chat_input[len("--no-memory ") :].strip()
                    status = (
                        "Retrieving past episodes and running support tools…"
                        if use_memory
                        else "Running support tools without episodic retrieval…"
                    )
                    with console.status(f"[cyan]{status}[/cyan]"):
                        result = await agent.handle(chat_input, use_memory=use_memory)
                    show_chat(
                        result.answer,
                        result.episode,
                        result.similar_episodes,
                        memory_enabled=result.memory_enabled,
                    )
                elif operation.startswith("/"):
                    show_error("Unknown command. Type /help to see available commands.")
                else:
                    show_error(
                        "Start with /chat or another command. Type /help for examples."
                    )
            except (EpisodeValidationError, EmbeddingError, AgentRunError) as error:
                show_error(str(error))
    finally:
        await database.close()
        show_goodbye()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
