"""Interactive CLI for persistent external KV memory."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex

from dotenv import load_dotenv

from agent import AgentRunError, DeveloperPreferenceAgent
from config import Settings
from memory import KVMemoryManager, MemoryValidationError
from models import JSONValue
from terminal_ui import (
    console,
    read_input,
    show_agent_answer,
    show_count,
    show_deleted,
    show_error,
    show_goodbye,
    show_help,
    show_memories,
    show_memory,
    show_start,
    show_stats,
    show_tool_event,
    show_usage,
)

EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}


def _parse_value(raw: str) -> JSONValue:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError as error:
        raise MemoryValidationError(f"Could not parse command: {error}") from error


async def run_cli() -> None:
    load_dotenv()
    try:
        settings = Settings.from_env()
        memory = KVMemoryManager(
            settings.database_path,
            max_ttl_seconds=settings.max_ttl_seconds,
        )
    except (ValueError, OSError) as error:
        show_error(f"Could not initialize KV memory: {error}")
        return

    agent = DeveloperPreferenceAgent(
        memory,
        settings,
        on_tool_event=show_tool_event,
    )
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

                if operation == "/set":
                    if len(parts) not in {5, 6}:
                        show_usage(
                            "/set",
                            "/set <namespace> <entity_id> <key> <json-value> [ttl_seconds]",
                            "/set project memory-series python_version 3.12",
                        )
                        continue
                    ttl = int(parts[5]) if len(parts) == 6 else None
                    item = memory.set(
                        parts[1], parts[2], parts[3], _parse_value(parts[4]), ttl_seconds=ttl
                    )
                    show_memory(item, title="Memory stored")
                    continue

                if operation == "/get":
                    if len(parts) != 4:
                        show_usage(
                            "/get",
                            "/get <namespace> <entity_id> <key>",
                            "/get project memory-series python_version",
                        )
                        continue
                    show_memory(memory.get(parts[1], parts[2], parts[3]))
                    continue

                if operation == "/delete":
                    if len(parts) != 4:
                        show_usage(
                            "/delete",
                            "/delete <namespace> <entity_id> <key>",
                            "/delete project memory-series python_version",
                        )
                        continue
                    show_deleted(memory.delete(parts[1], parts[2], parts[3]))
                    continue

                if operation == "/list":
                    if len(parts) != 3:
                        show_usage(
                            "/list",
                            "/list <namespace> <entity_id>",
                            "/list project memory-series",
                        )
                        continue
                    show_memories(memory.list(parts[1], parts[2]), parts[1], parts[2])
                    continue

                if operation == "/clear":
                    if len(parts) not in {2, 3}:
                        show_usage(
                            "/clear",
                            "/clear <namespace> [entity_id]",
                            "/clear project memory-series",
                        )
                        continue
                    count = memory.clear_namespace(parts[1], parts[2] if len(parts) == 3 else None)
                    show_count("Memories deleted", count)
                    continue

                if operation == "/cleanup":
                    if len(parts) != 1:
                        show_usage("/cleanup", "/cleanup", "/cleanup")
                        continue
                    show_count("Expired rows deleted", memory.cleanup_expired())
                    continue

                if operation == "/stats":
                    if len(parts) != 1:
                        show_usage("/stats", "/stats", "/stats")
                        continue
                    show_stats(memory.stats(), settings.database_path)
                    continue

                if operation == "/chat":
                    chat_parts = command.split(maxsplit=4)
                    if len(chat_parts) != 5:
                        show_usage(
                            "/chat",
                            "/chat <namespace> <entity_id> <key1,key2> <message>",
                            "/chat project memory-series python_version,test_framework What stack should we use?",
                        )
                        continue
                    keys = [key.strip() for key in chat_parts[3].split(",") if key.strip()]
                    if not keys:
                        raise MemoryValidationError("/chat requires at least one exact key")
                    with console.status("[cyan]Retrieving exact keys and calling OpenAI…[/cyan]"):
                        answer = await agent.chat_with_known_memories(
                            chat_parts[4],
                            namespace=chat_parts[1],
                            entity_id=chat_parts[2],
                            keys=keys,
                        )
                    show_agent_answer(answer, "Application-managed KV retrieval")
                    continue

                if operation == "/agent":
                    agent_parts = command.split(maxsplit=1)
                    if len(agent_parts) != 2 or not agent_parts[1].strip():
                        show_usage(
                            "/agent",
                            "/agent <message>",
                            "/agent For project memory-series, always use Python 3.12.",
                        )
                        continue
                    with console.status("[cyan]Calling OpenAI with validated KV tools…[/cyan]"):
                        answer = await agent.run_tool_agent(agent_parts[1])
                    show_agent_answer(answer, "Tool-managed KV agent")
                    continue

                if operation.startswith("/"):
                    show_error("Unknown command. Type /help to see available commands.")
                else:
                    show_error(
                        "Choose an execution mode: use /chat for application-managed exact "
                        "retrieval or /agent for model-requested KV tools. Type /help for examples."
                    )
            except (MemoryValidationError, ValueError) as error:
                show_error(str(error))
            except AgentRunError as error:
                show_error(str(error))
    finally:
        memory.close()
        show_goodbye()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
