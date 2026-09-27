# Memory #3 — Vector / Semantic Memory

A production-style learning project for persistent memory retrieved by meaning. The storage layer uses local ChromaDB, embeddings come through a small provider interface, and optional chat uses the OpenAI Responses API.

```text
                    VECTOR MEMORY

User explicitly remembers engineering knowledge
                       ↓
                embedding provider
                       ↓
          local persistent ChromaDB
                       ↓
new query → embedding → nearest memories → application → model
```

Vector memory is external memory. It survives restarts, but it is neither the conversation transcript nor the model’s context window. Ordinary `/chat` messages are never stored automatically.

## What vector memory is for

Vector memory is useful when the application knows the meaning it needs but not an exact key:

- similar incidents and root causes
- engineering observations and learnings
- preferences expressed in different words
- facts that users may ask for with varied phrasing

For exact configuration such as `project / memory-series / python_version`, use KV memory. Semantic search ranks candidates and can be imperfect; exact key lookup is deterministic.

## Architecture

Writing is explicit:

```text
/remember <text>
      ↓
validate + normalize-content dedupe
      ↓
EmbeddingProvider.embed(text)
      ↓
VectorMemoryManager.add_memory(...)
      ↓
local ChromaDB collection: agent_memory
```

Retrieval is application managed:

```text
/chat <question>
      ↓
embed question → top-k Chroma search
      ↓
typed SearchResult objects
      ↓
format as UNTRUSTED RETRIEVED MEMORY DATA
      ↓
OpenAI Responses API
```

`VectorMemoryManager` is the application boundary. Callers do not handle Chroma collections, query response shapes, storage metadata, or distance conversion. `EmbeddingProvider` keeps the OpenAI API separate from storage and lets the test suite use deterministic offline embeddings.

## Data model

Each memory contains:

- stable UUID
- content
- type: `observation`, `fact`, `preference`, `incident`, or `learning`
- source
- creation and update timestamps
- JSON metadata
- importance from 1 to 5

Search returns the memory, Chroma cosine distance, and `1 - distance` as a convenient similarity score. The score is useful for ranking within this collection. It is not a probability, confidence, or universal quality measure.

## Lifecycle and deduplication

Vector memory does not use token-based eviction.

- **Add:** `/remember` creates an embedding and persists the memory.
- **Deduplicate:** identical normalized content (case-folded with whitespace collapsed) reuses the existing memory.
- **Update:** the manager can update attributes and re-embeds only when content changes.
- **Delete:** `/delete <id>` removes a memory explicitly.
- **Clear:** `VectorMemoryManager.clear()` supports deliberate administrative cleanup.
- **Retention:** a production application should add domain-specific retention, archival, and deletion rules.

## Memory types compared

| Property | Working memory | KV memory | Vector memory |
|---|---|---|---|
| Main purpose | Current execution context | Exact durable facts | Relevant durable knowledge |
| Retrieval | Context assembly | Namespace + entity + key | Semantic similarity |
| Storage in this series | In-process items + summary | SQLite | ChromaDB |
| Typical question | “What happened earlier in this run?” | “What is this project’s Python version?” | “Have we seen a similar incident?” |
| Main lifecycle pressure | Token budget | TTL, overwrite, delete | Dedupe, retention, update, delete |
| Retrieval certainty | Included or omitted | Deterministic | Ranked and approximate |

## Setup

Use Python 3.10 or newer and a project-local virtual environment:

```bash
cd memory/03-vector-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to use `/remember`, `/search`, or `/chat`. The first two require the embeddings API; `/chat` additionally calls the Responses API. Local `/list`, `/memory`, `/delete`, and `/stats` work without an API call after memories exist.

Default settings:

```dotenv
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PATH=.data/chroma
CHROMA_COLLECTION=agent_memory
VECTOR_TOP_K=5
VECTOR_MAX_TOP_K=25
VECTOR_MIN_SIMILARITY=0.20
```

The `.env` file and `.data/` directory are ignored by Git.

## Run

```bash
python main.py
```

The Rich terminal UI shows the active configuration, commands, progress, colored result tables, and the similarity-score caveat.

Exact demo flow:

```text
/remember --type incident --source incident-2026-09 checkout latency was caused by fraud-service CPU saturation
/remember --type learning --source postmortem checkout timeouts can originate in a downstream fraud dependency
/remember --type preference --importance 4 use Python 3.12 for the memory series
/search what caused the checkout slowdown?
/search --top-k 2 --type incident why was checkout slow?
/list
/stats
/chat Have we seen a similar checkout incident, and what should I inspect?
/quit
```

`/remember <text>` defaults to type `observation`, source `cli`, and importance `3`. Quote text when your shell-like command contains special spacing or when an option value must stay together.

## Filtering

The manager supports `top_k`, a practical maximum top-k, optional minimum similarity, and filters for type, source, importance, or scalar user metadata. `/chat` applies `VECTOR_MIN_SIMILARITY` (default `0.20`) before sending memories to the model, while `/search` accepts an explicit threshold for exploration. The CLI exposes the most useful controls:

```text
/search --top-k 3 --type incident --min-score 0.25 checkout timeout
```

Filtering narrows candidates; it does not turn semantic ranking into an exact guarantee. Calibrate the default chat threshold against representative queries before production use.

## Security boundaries

Stored content is untrusted data. A memory may contain stale facts, malicious prompt text, accidental secrets, or instructions copied from an external source.

The application therefore:

- labels retrieved records as untrusted data
- sends them in a user-role data envelope, never as developer instructions
- keeps behavioral rules in the separate trusted `instructions` field
- validates text length, type, source, importance, metadata, top-k, and score thresholds
- stores only through explicit `/remember`
- logs IDs, event types, result counts, and latency without logging memory content
- keeps the API key and local Chroma files out of Git

A production system should also enforce tenant isolation, authorization on every read and write, encryption, retention policies, audit controls, deletion workflows, input classification, and evaluation of retrieval quality.

## Testing

Tests use a deterministic `FakeEmbeddingProvider` and temporary Chroma directories. They do not call OpenAI and do not require an API key.

```bash
python -m pytest -q
```

The suite covers persistence, normalized-content deduplication, ranking, filters, thresholds, update and delete lifecycle, validation, offline embedding substitution, application-managed retrieval, prompt-injection isolation, and the rule that chat is never stored automatically.

## Production evolution

For a real service, keep the same interfaces and replace deployment details deliberately:

- batch embedding calls and add retry/backoff with bounded timeouts
- pin and version embedding models; re-embed through a migration job when dimensions or models change
- add tenant and authorization filters before search
- evaluate recall and ranking against a labeled query set
- calibrate score thresholds per embedding model and corpus
- add backups, retention jobs, deletion guarantees, and collection migrations
- move Chroma to an appropriately operated service only when scale or availability requires it
- trace retrieval IDs and latency while keeping stored content out of logs

The core lesson remains visible: vector memory is an application-owned retrieval system, not automatic model memory.
