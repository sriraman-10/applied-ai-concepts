# Memory #2 — External Key-Value Memory

A production-style learning project for persistent, deterministic agent memory. The storage layer uses SQLite, while optional agent execution uses `AsyncOpenAI` and the Responses API.

```text
                     KV MEMORY

Agent knows exact information needed
              ↓
 namespace + entity + key
              ↓
            SQLite
              ↓
        deterministic value
```

```text
(user, 123, preferred_language)
              ↓
            "Java"
```

KV memory is separate from chat history. Local `/set`, `/get`, `/list`, `/delete`, `/clear`, `/cleanup`, and `/stats` commands do not call OpenAI and work without an API key.

## What KV memory is for

KV memory is useful when:

- the key is known
- deterministic lookup is desirable
- data is structured
- exact correctness matters
- information must survive a process restart

Examples include user preferences, project settings, feature flags, service ownership, stable identifiers, and environment configuration.

KV memory is not appropriate for “Have we seen something similar to this incident before?” That requires semantic retrieval and belongs in **Memory #3 — Vector Memory**.

## Architecture

Writing:

```text
Explicit user preference or application event
                  ↓
     namespace + entity + key + value
                  ↓
        KVMemoryManager.set(...)
                  ↓
                SQLite
```

Reading:

```text
Agent/application knows the required fact
                  ↓
   get(namespace, entity_id, key)
                  ↓
         indexed exact lookup
                  ↓
        typed JSON value or None
```

The logical unique key is `(namespace, entity_id, key)`. These values remain separate SQL parameters; the implementation never constructs a compound key through unsafe string concatenation.

## Data model

Each entry contains:

- stable UUID
- namespace
- entity ID
- key
- JSON value
- value type
- creation and update timestamps
- optional expiration timestamp
- optional JSON metadata

Values can be strings, numbers, booleans, lists, objects, or null. SQLite stores canonical JSON plus an explicit value type so round trips are predictable.

## Lifecycle and eviction

KV memory does not use token-based eviction.

- **Overwrite:** `set()` updates an existing logical key while preserving its ID and `created_at`.
- **Explicit delete:** `delete()` removes one key.
- **TTL:** an optional positive lifetime makes temporary entries invisible after expiration.
- **Cleanup:** `cleanup_expired()` physically removes expired rows.
- **Scope cleanup:** `clear_namespace()` clears a whole namespace or one entity.
- **Business rules:** application code decides when a durable fact should be replaced or removed.

An expired entry behaves as missing immediately, even before physical cleanup.

## Two agent integrations

### A. Application-managed memory

`/chat` requires the caller to name the exact keys:

```text
Application
    ↓
retrieve selected exact keys
    ↓
inject only those values
    ↓
Responses API
```

Example:

```text
/chat project memory-series python_version,test_framework What stack should we use?
```

This is the preferred path when application code already knows which configuration fields are relevant.

### B. Tool-managed memory

`/agent` supplies four validated Responses API tools:

- `get_memory`
- `set_memory`
- `delete_memory`
- `list_memories`

The model can request those operations, but it never receives a SQL interface. Every argument passes through `KVMemoryManager` validation and parameterized SQL.

Because `/agent` is specifically the tool-managed mode, its first model turn must call a KV tool. Later turns return to automatic tool selection so the model can produce a final answer after observing the database result. The prompt also defines the address convention: namespaces use lowercase words, entity IDs use kebab-case, and keys use snake_case.

```text
/agent For project memory-series, always use Python 3.12.
/agent What Python version should project memory-series use?
```

The write policy is intentionally narrow: store only an explicit durable preference or exact configuration. Greetings, raw logs, arbitrary chat, temporary conversation, and model reasoning are never persisted.

## Working memory versus KV memory

| Property | Working memory | KV memory |
|---|---|---|
| Purpose | Current execution context | Exact persistent facts |
| Storage | Model context/application | SQLite, Redis, DynamoDB, etc. |
| Retrieval | Context assembly | Exact namespace + entity + key |
| Eviction | Token pressure and compaction | TTL, delete, replacement, business rule |
| Persistence | Usually temporary | Yes |

## Why not ChromaDB?

Embeddings and semantic search add cost, latency, ranking behavior, and operational complexity. They are unnecessary when the application knows that it needs `project / memory-series / python_version`. Exact indexed lookup is simpler and more correct for that access pattern.

## Setup

Use Python 3.10 or newer and a project-local virtual environment:

```bash
cd /Users/sriraman10/Documents/blog/applied-ai-concepts/memory/02-kv-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you want to use `/chat` or `/agent`. CRUD commands require no API key. The default database is `.data/kv_memory.db`; both the database and its SQLite sidecar files are ignored by Git.

## Run

```bash
python main.py
```

The colored startup screen explains every command. JSON values can be typed directly. Non-JSON tokens are stored as strings.

```text
/set project memory-series python_version 3.12
/set project memory-series test_framework pytest
/set user 123 preferences '{"theme":"dark","compact":true}'
/set deployment release-42 region '"ap-south-1"' 3600
/get project memory-series python_version
/list project memory-series
/delete project memory-series test_framework
/clear project memory-series
/cleanup
/stats
```

Because `3.12` is valid JSON, the first example stores a number. Use `'"3.12"'` when the distinction between a version string and a number matters.

## Persistence demonstration

First run:

```text
/set project memory-series python_version '"3.12"'
/get project memory-series python_version
/quit
```

Restart with `python main.py`, then run:

```text
/get project memory-series python_version
```

The same value and original `created_at` are loaded from SQLite. No conversation transcript is needed.

## Observability and safety

The manager emits structured lifecycle logs for reads, writes, updates, deletes, expiration, cleanup, and scope clearing. Logs include the address and outcome, not the stored value.

Other safeguards:

- parameterized SQL only
- indexed namespace/entity lookup
- unique database constraint on the logical key
- bounded identifier lengths and no control characters
- validated positive TTL with a configurable maximum
- finite JSON numbers only
- `.env`, SQLite files, caches, and virtual environments ignored

## Tests

Tests use isolated temporary SQLite files and fake OpenAI clients. They make no network calls and require no API key.

```bash
python -m pytest -q
```

Coverage includes persistence, updates, timestamp semantics, deletion, exact lookup, namespace/entity isolation, TTL, cleanup, JSON values, duplicate keys, SQL-injection-like strings, scoped clearing, application-managed context injection, and tool-managed writes.

## Project layout

```text
02-kv-memory/
├── main.py             # colored CLI and command parsing
├── agent.py            # Responses API integrations and validated tools
├── memory.py           # SQLite KV manager and lifecycle policy
├── models.py           # typed values and records
├── config.py           # environment settings
├── terminal_ui.py      # Rich panels and tables
├── tests/              # isolated offline tests
└── requirements.txt
```

## Production evolution

The storage backend can evolve while preserving the manager contract:

```text
SQLite
  ↓
Redis / DynamoDB / Postgres
```

The same `KVMemoryManager` boundary can sit underneath the normal OpenAI SDK, OpenAI Agents SDK, or another orchestration framework. Memory addressing, validation, lifecycle, and access control remain independent of the framework that decides when to read or write.

Vector, episodic, and procedural memory are deliberately outside this project.
