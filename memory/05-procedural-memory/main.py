"""Interactive CLI for procedural memory."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from dotenv import load_dotenv

from agent import ProceduralSupportAgent
from config import Settings
from database import PostgresDatabase
from embeddings import OpenAIEmbeddingProvider
from memory import ProceduralMemoryManager
from procedure_executor import ProcedureExecutor
from seeds.seed_procedures import seed_procedures
from store import PostgresProcedureStore
from terminal_ui import *
from tools import SupportTools


async def run() -> None:
    load_dotenv()
    settings = Settings.from_env()
    if not settings.migrations_path.is_absolute():
        settings = replace(
            settings, migrations_path=Path(__file__).parent / settings.migrations_path
        )
    database = PostgresDatabase(
        settings.database_url, settings.embedding_dimensions, settings.migrations_path
    )
    await database.open()
    store = PostgresProcedureStore(database)
    manager = ProceduralMemoryManager(
        store,
        OpenAIEmbeddingProvider(
            settings.embedding_model, settings.embedding_dimensions
        ),
        max_top_k=settings.max_top_k,
    )
    tools = SupportTools()
    executor = ProcedureExecutor(store, tools, on_tool_event=show_tool_event)
    agent = ProceduralSupportAgent(manager, executor, settings)
    show_start(settings)
    try:
        while True:
            raw = read_input()
            if not raw:
                continue
            if raw in {"/quit", "exit", "quit"}:
                break
            try:
                command, _, arg = raw.partition(" ")
                if command == "/help":
                    show_help()
                elif command == "/seed":
                    show_info(
                        f"Created {len(await seed_procedures(manager))} procedure(s); existing versions were left unchanged.",
                        "Seed complete",
                    )
                elif command == "/procedures":
                    show_procedures(await manager.list_procedures())
                elif command == "/procedure":
                    parts = arg.split()
                    if not parts:
                        raise ValueError("Usage: /procedure <key> [version]")
                    item = (
                        await manager.get_procedure(parts[0], int(parts[1]))
                        if len(parts) > 1
                        else await manager.get_active_version(parts[0])
                    )
                    show_procedure(item)
                elif command == "/search":
                    show_search(
                        await manager.search_procedures(
                            arg,
                            top_k=settings.top_k,
                            min_similarity=settings.min_similarity,
                        )
                    )
                elif command in {"/activate", "/deactivate"}:
                    key, version = arg.split()
                    fn = (
                        manager.activate_procedure
                        if command == "/activate"
                        else manager.deactivate_procedure
                    )
                    p = await fn(key, int(version))
                    show_info(
                        f"{p.procedure_key} v{p.version} is {p.status.value}.",
                        "Procedure status",
                    )
                elif command == "/executions":
                    show_executions(await store.list_executions())
                elif command == "/execution":
                    show_execution(await store.get_execution(arg.strip()))
                elif command == "/chat":
                    show_chat(await agent.handle(arg))
                elif command == "/stats":
                    show_stats(await manager.count())
                else:
                    raise ValueError("Unknown command. Type /help.")
            except Exception as error:
                show_error(str(error))
    finally:
        await database.close()
        show_goodbye()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(run())
