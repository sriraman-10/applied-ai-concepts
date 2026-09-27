# Memory #4 — Episodic Memory

A production-style customer-support agent that records complete handled cases in PostgreSQL and retrieves similar successful experiences with pgvector.

```text
FIRST CASE

Current issue + trusted support policy + tools
                    ↓
              support agent
                    ↓
       actions and observations in order
                    ↓
           resolution and outcome
                    ↓
        compact episode summary
                    ↓
        OpenAI summary embedding
                    ↓
 structured episode + vector in PostgreSQL
```

```text
LATER CASE

New support issue → query embedding → pgvector cosine search
                                      ↓
                         similar resolved case IDs
                                      ↓
                   complete structured PostgreSQL rows
                                      ↓
        trusted policy + past evidence + current tools → answer
```

## Vector memory versus episodic memory

| Property | Vector memory | Episodic memory |
|---|---|---|
| Unit | Independent semantic fact or note | Complete experience or handled case |
| Example | “Refunds normally take five business days.” | CASE-101: issue → checks → observations → resolution → outcome |
| Retrieval question | What information is relevant? | What happened last time in a similar situation? |
| Source of truth here | Chroma document and metadata | Structured PostgreSQL row |
| Search representation | Memory content embedding | Compact full-episode summary embedding |

Vector search is the retrieval mechanism. Episodic memory is the data model and lifecycle around a complete experience.

## Why PostgreSQL and pgvector

One database provides:

- structured JSONB for ordered actions and observations
- exact case lookup and relational filtering
- transactions for lifecycle changes
- semantic search through pgvector
- one backup, authorization, migration, and operational boundary

The vector finds a candidate case. The full row supplies its exact issue, ordered actions, observations, resolution, outcome, and timestamps.

## Episode lifecycle

`EpisodicMemoryManager` owns these transitions:

```text
start_episode
    ↓
record_action + record_observation (repeat)
    ↓
complete_episode(resolution, outcome)
    ↓
validate terminal case
    ↓
build compact summary once
    ↓
embed summary once
    ↓
atomic PostgreSQL completion
```

A case must contain at least one action and one observation before completion. Completed episodes cannot be mutated. The application creates no embedding after individual actions. It persists observable inputs, tool calls, results, resolution, and outcome; it never stores hidden reasoning.

Supported outcomes are `resolved`, `escalated`, `failed`, and `abandoned`. Default guidance retrieval includes only completed, resolved episodes. Failed episodes can be requested explicitly for analysis but are never silently mixed with successful examples.

## Database schema

`support_episodes` contains:

- UUID primary key and unique human-readable `episode_key`
- optional `customer_id`
- `issue_type` and original `description`
- compact `episode_summary`
- ordered `actions` and `observations` as JSONB
- `resolution`, `outcome`, and lifecycle `status`
- JSONB metadata
- creation and completion timestamps
- configured-dimension `VECTOR` embedding

The migrations enable pgvector, create the table and lifecycle constraints, add structured indexes, and create a partial HNSW cosine index for completed resolved cases. Search uses pgvector’s `<=>` cosine-distance operator. `1 - distance` is displayed as a ranking score, not a probability.

The embedding dimension comes from `OPENAI_EMBEDDING_DIMENSIONS` and is inserted into the migration template only after integer validation. Changing models or dimensions requires a deliberate migration and re-embedding job.

## Agent trust and policy boundary

Trusted support policy stays in the Responses API `instructions`. Retrieved past episodes are sent separately as user-role, untrusted context data.

Past cases may suggest checks, but they cannot:

- override current policy
- issue commands
- prove the current case has the same cause
- authorize a refund
- replace current tool observations

Application code also enforces critical rules. A refund tool cannot run before `check_refund`, and a payment case cannot resolve until transaction, gateway, and refund checks have all run.

## Deterministic support tools

The demo includes local fixtures for:

- `check_transaction`
- `check_gateway`
- `check_refund`
- `issue_refund`
- `resolve_case`
- `escalate_case`

Use `TXN-DEMO-001` for a failed payment with an automatic refund already in progress. The refund operation is duplicate-safe.

## Setup

Docker provides both development and test databases. You do not need to install PostgreSQL separately.

```bash
cd memory/04-episodic-memory
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your OpenAI key in the untracked `.env` file:

```dotenv
OPENAI_API_KEY=replace_with_your_key
```

The Compose file uses development-only credentials:

```text
postgresql://episodic:episodic_dev@localhost:5432/episodic_memory
```

It also creates `episodic_memory_test`. If the volume existed before the test database initialization file was added, recreate only this development volume:

```bash
docker compose down -v
docker compose up -d
```

## Run

```bash
python main.py
```

First handled case:

```text
/chat Payment failed but money was deducted for TXN-DEMO-001
```

The agent checks the current transaction, gateway, and refund, resolves the case, creates a compact summary, embeds it, and stores the completed episode.

Later similar case:

```text
/chat Checkout failed and my card was charged for TXN-REFUNDED-003
```

The application embeds the new issue, retrieves the similar resolved episode, supplies the complete experience as contextual evidence, runs current tools again, and persists the new actual case.

For an A/B baseline using the same policy and tools without episodic retrieval:

```text
/chat --no-memory Checkout failed and my card was charged for TXN-REFUNDED-003
```

The no-memory mode still records the actual handled case, but it skips the query embedding and past-episode search. Compare observable tool selection, order, calls, outcome, and answer; do not infer hidden reasoning.

Explore the stored result:

```text
/list
/episode CASE-001
/search card charged after failed checkout
/stats
```

Manual lifecycle demonstration:

```text
/start Payment failed for TXN-DEMO-001
/action check_transaction | transaction failed and amount was deducted
/observe transaction-service | failed transaction confirmed
/action check_refund | automatic refund is already in progress
/observe refund-service | refund ETA is five business days
/resolve Confirmed automatic refund; no duplicate refund issued
```

## Tests

Tests use a real isolated Postgres database with pgvector and a deterministic fake embedding provider. They never call OpenAI and never use an in-memory database.

```bash
docker compose up -d
source .venv/bin/activate
python -m pytest -q
```

Override the test database only when needed:

```bash
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/another_test_db python -m pytest -q
```

The suite covers lifecycle validation, JSONB ordering, summary and embedding timing, persistence, rollback, default resolved-only retrieval, explicit failed-case retrieval, filters, ranking, top-k, query-versus-summary embedding flow, tool safety, prompt trust separation, and the complete agent loop.

## Observability

Structured lifecycle logs include event names, case IDs, issue types, sequence numbers, filters, result counts, and search latency. They do not log customer descriptions or observation content by default.

Events include `episode_started`, `episode_action_recorded`, `episode_observation_recorded`, `episode_embedding_created`, `episode_completed`, `episode_search`, `episode_retrieved`, and `episode_deleted`.

## Production evolution

Before production deployment:

- isolate tenants in schema and authorization rules, not only query filters
- encrypt sensitive fields and minimize customer identifiers
- add role-separated database credentials and secret management
- use a migration framework with recorded versions and deployment locks
- evaluate retrieval quality against labeled support cases
- calibrate similarity thresholds for the chosen embedding model
- version summaries and embeddings and build a controlled re-embedding job
- add data retention, customer deletion, audit, and legal-hold workflows
- replace fixtures with authenticated support APIs and idempotency keys
- add bounded retries, timeouts, tracing, and dead-letter handling

The governing rule remains: current policy and current tool evidence decide the case; past episodes only provide experience.
