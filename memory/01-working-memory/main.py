"""Interactive CLI for the working-memory demonstration."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from agent import AgentRunError, WorkingMemoryAgent
from config import Settings
from terminal_ui import (
    console,
    read_input,
    show_assistant,
    show_compaction,
    show_error,
    show_goodbye,
    show_help,
    show_memory,
    show_start,
    show_stats,
    show_summary,
    show_tool_added,
    show_usage_error,
)

EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}


async def run_cli() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        show_error("Set OPENAI_API_KEY in the environment or in .env before running.")
        return

    try:
        settings = Settings.from_env()
    except ValueError as error:
        show_error(f"Invalid configuration: {error}")
        return

    agent = WorkingMemoryAgent(settings, on_compaction=show_compaction)
    show_start(settings)

    while True:
        try:
            command = await asyncio.to_thread(read_input)
        except (EOFError, KeyboardInterrupt):
            console.print()
            show_goodbye()
            return
        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            show_goodbye()
            return
        if command == "/stats":
            show_stats(agent.memory.stats, settings)
            continue
        if command == "/memory":
            show_memory(agent.memory.items)
            continue
        if command == "/summary":
            show_summary(agent.memory.summary)
            continue
        if command == "/help":
            show_help()
            continue
        if command.startswith("/tool"):
            parts = command.split(maxsplit=2)
            if len(parts) != 3 or parts[1] not in {"important", "noise"}:
                show_usage_error()
                continue
            important = parts[1] == "important"
            try:
                await agent.add_tool_observation(parts[2], important=important)
            except ValueError as error:
                show_error(f"Could not add observation: {error}")
            else:
                show_tool_added(important=important, content=parts[2])
            continue
        if command.startswith("/"):
            show_error("Unknown command. Type /help to see the available commands.")
            continue

        try:
            with console.status("[cyan]Calling OpenAI and updating working memory…[/cyan]"):
                answer = await agent.chat(command)
        except AgentRunError as error:
            show_error(str(error))
        else:
            show_assistant(answer)


def main() -> None:
    # Keep dependency transport logs quiet. Application errors and explicit
    # compaction panels remain visible to the learner.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
