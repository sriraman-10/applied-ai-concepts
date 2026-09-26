# Memory #1 — Working / In-Context Memory

A small Python agent that owns its working-memory policy instead of delegating conversation history to an agent framework. It uses `AsyncOpenAI` and the Responses API for replies and compaction summaries.

```text
Working Memory
      ↓
Context grows
      ↓
Token limit
      ↓
Eviction
 ├── drop noise
 ├── summarize old useful state
 └── preserve recent state
      ↓
LLM
```

## What this project teaches

Working memory is the bounded context assembled for the next model request. Every item has an ID, role, content, kind, priority, UTC timestamp, and estimated token count. The application decides what remains visible to the model.

This project implements **working memory only**. It is not:

- KV memory
- vector memory
- episodic memory
- procedural memory

It does not use Agents SDK Sessions, LangGraph, Mem0, a vector database, Redis, or another memory framework.

## Architecture

```text
User
  ↓
WorkingMemoryAgent
  ↓
MemoryManager.add(...)
  ↓
Pre-flight token estimate with tiktoken
  ↓
At the hard limit
  ├── 1. Remove old low-priority tool noise
  ├── 2. Merge older useful context into the existing summary
  └── 3. Hard-evict oldest context only when still necessary
  ↓
MemoryManager.build_model_input()
  ↓
OpenAI Responses API
  ↓
Store assistant response in working memory
```

The limits have separate purposes:

- `MEMORY_HARD_LIMIT` triggers compaction.
- `MEMORY_TARGET_LIMIT` gives the conversation room to grow before compaction runs again.
- `MEMORY_KEEP_RECENT` protects the newest items during noise removal and summarization.

Recent entries can still be removed in the final safety phase when one unusually large message cannot fit within the target. The summary represents older context, so an oversized summary is trimmed before protected recent entries are discarded.

## Memory kinds

- `USER_MESSAGE`: observable user input
- `ASSISTANT_MESSAGE`: model output
- `TOOL_OBSERVATION`: simulated or future real tool results
- `DECISION`: an explicit application or agent decision

Tool observations use explicit priority. Low-priority noise is the first eviction candidate; important observations are summarized with other useful history.

## Summary contract

The summary call can retain only:

- current objective
- confirmed facts
- important observations
- constraints
- decisions
- actions already attempted
- unresolved questions

It removes greetings, repetition, verbose logs, irrelevant tool output, and obsolete intermediate details. Each later compaction sends both the current summary and newly evicted history to the summarizer, so the model merges them rather than replacing old state blindly.

The summary is observable compressed state. It does not contain or request private chain-of-thought.

## Why implement this manually?

Production frameworks can store session history and compact context automatically. This project deliberately owns the policy so the code makes every decision visible: how tokens are estimated, which observations count as noise, which recent items are protected, what is summarized, and when hard eviction occurs. That control makes memory behavior testable and prevents an opaque session abstraction from deciding what reaches the model.

## Setup

Use Python 3.10 or newer and a project-local environment:

```bash
cd memory/01-working-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the placeholder key. `.env` is ignored by Git and loaded by `python-dotenv`.

For a short demonstration, the provided example uses:

```dotenv
MEMORY_HARD_LIMIT=1500
MEMORY_TARGET_LIMIT=900
MEMORY_KEEP_RECENT=4
```

These are deliberately small learning values. Real limits should reserve enough room for instructions, tool schemas, reasoning, and output tokens in addition to the working-memory estimate.

`tiktoken` may download its encoding table on first use. If that table is unavailable, the estimator logs a warning and safely overestimates at one token per UTF-8 byte instead of preventing the agent from starting.

## Run

```bash
python main.py
```

The startup screen explains the two input modes: a normal sentence executes the agent and calls OpenAI; a slash command inspects or changes local memory without a model call. Output uses colored panels and tables, and routine HTTP transport logs are hidden.

Commands:

```text
Any normal sentence executes the agent
/stats
/memory
/summary
/tool important checkout-service p95 latency is 1850ms
/tool noise health-check endpoint returned 200
/quit (exit, quit, and /exit also work)
```

The same tool-observation API is available in Python:

```python
await agent.add_tool_observation(
    "checkout-service p95 latency is 1850ms",
    important=True,
)
await agent.add_tool_observation(
    "health-check endpoint returned 200",
    important=False,
)
```

`/stats` reports estimated tokens, item count, summary tokens, whether a summary exists, and the compaction count. Compaction also emits a structured log line containing `before_tokens`, `after_tokens`, `items_dropped`, `items_summarized`, and `compaction_count`.

## Test without an API key

```bash
python -m pytest -q
```

Tests inject a deterministic word estimator and fake async summarizer. They make no network calls and require no API key.

## Project layout

```text
01-working-memory/
├── main.py             # interactive CLI
├── agent.py            # Responses API execution and LLM summarizer
├── memory.py           # storage and three-phase memory policy
├── token_estimator.py  # tiktoken adapter
├── config.py           # validated environment settings
├── tests/              # offline policy tests
└── requirements.txt
```

## Production evolution

The `MemoryManager` is independent of the Responses API loop. A later project can place the same interface beneath an OpenAI Agents SDK agent while the SDK handles orchestration. Persistence, tenant isolation, encryption, distributed locking, summary evaluation, and durable storage would also be added before using this design across production processes.
