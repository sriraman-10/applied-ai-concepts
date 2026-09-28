# Memory #5 — Procedural Memory

A production-style customer-support demo that answers:

> How should this task be performed?

Procedural memory stores approved, reusable playbooks. It does not store a transcript of one past case. This project retrieves the relevant active playbook with pgvector, loads its structured steps, executes those steps through deterministic support tools, and records the run separately in PostgreSQL.

```text
User issue
    ↓
query embedding
    ↓
PostgreSQL + pgvector
    ↓
ranked ACTIVE procedures
    ↓
highest relevant approved playbook
    ↓
structured steps + application safety
    ↓
support tools
    ↓
completed or escalated execution
    ↓
grounded Responses API answer
```

## What the demo makes visible

- Human/admin-owned procedure definitions
- Compact searchable summaries rather than embedded JSON
- Active-only semantic retrieval with a minimum similarity threshold
- Explicit procedure versions and approval states
- One active version per procedure key
- Immutable procedure definitions during execution
- Separate execution and step-execution records
- Application-enforced tool order and refund idempotency
- Ranked candidates, selected procedure, completed steps, and outcome in the terminal
- Trusted procedures separated from untrusted user input and tool data

## Memory types in the series

| Memory | Question |
|---|---|
| Working memory | What is happening now? |
| KV memory | What exact fact do I know? |
| Vector memory | What semantic knowledge is relevant? |
| Episodic memory | What happened in a similar past case? |
| Procedural memory | How should this task be performed? |

An episode records what actually happened in `CASE-101`: its actions, observations, and outcome. A procedure records what an approved agent should do every time a matching task occurs.

```text
EPISODIC                              PROCEDURAL
CASE-101                              FAILED_PAYMENT_PLAYBOOK v1
what happened before?                 what should I do?
actions actually taken                approved ordered steps
case-specific outcome                 reusable completion rules
```

Successful episodes can inform a future procedure, but they must not rewrite production policy automatically:

```text
many successful episodes
        ↓
offline analysis
        ↓
human review
        ↓
explicit new procedure version
```

## System instructions versus procedures

System instructions contain global invariants, such as never revealing private customer information. Procedural memory contains task-specific operational knowledge, such as checking transaction, gateway, and refund state for a failed payment.

Putting every playbook permanently in one system prompt makes approval, versioning, audit, retrieval, and maintenance difficult. This project retrieves only the relevant approved playbook. That keeps prompts smaller and makes each production change explicit and reviewable.

## Stored procedures

`/seed` creates three intentionally distinct active playbooks:

1. `FAILED_PAYMENT_PLAYBOOK`
2. `DUPLICATE_CHARGE_PLAYBOOK`
3. `REFUND_DELAY_PLAYBOOK`

The embedded text is a compact summary containing the name, description, category, issue patterns, important actions, and escalation concepts. The JSONB steps remain the source of truth for execution.

## Schema

`support_procedures` stores persistent operational knowledge:

- identity: `id`, `procedure_key`, `version`
- descriptive fields: `name`, `description`, `category`, `procedure_summary`
- lifecycle: `status`, `created_at`, `updated_at`
- source of truth: `steps`, `preconditions`, `completion_conditions`, `escalation_rules`
- extension data: `metadata`
- retrieval: `embedding`

The database enforces unique `(procedure_key, version)` rows and a partial unique index allowing at most one active version per key.

`procedure_executions` records one run against an exact procedure ID, key, and version. `procedure_step_executions` records each ordered step, its tool, status, result summary, and timestamps. Execution never modifies the procedure row.

## Version lifecycle

```text
create v1 → draft → activate
                         ↓
create v2 → draft → activate v2
                         ↓
                v1 becomes deprecated
```

`create_new_version()` copies the latest definition into a new draft version and recomputes its searchable summary and embedding. It never overwrites the earlier row. Activation runs in a transaction: any previous active version becomes deprecated before the selected version becomes active. A draft can be returned to editing with deactivation; deprecation preserves it for audit.

Only draft versions may be deleted. Active and deprecated procedures remain auditable.

## Retrieval

`search_procedures(query, top_k=3, category=None, min_similarity=None)` embeds the issue and performs cosine-distance search using pgvector's `<=>` operator. Results use `1 - cosine_distance` for ranking; that score is not a probability.

The HNSW index uses `vector_cosine_ops` and is partial on `status = 'active'`, matching the default production query. Draft and deprecated rows cannot be selected by the agent. When no result reaches the configured threshold, `/chat` performs no procedure execution and returns the configured support/escalation fallback.

This project deliberately selects one highest relevant playbook. Multi-procedure composition is outside its scope.

## Execution and safety

The application owns procedure selection and step execution. The LLM writes the final grounded customer response after execution; it cannot administer playbooks, skip the active-status check, call arbitrary SQL, or bypass the executor.

Trust boundaries are explicit:

| Input | Trust |
|---|---|
| System/developer instructions | Trusted global rules |
| Approved active procedure | Trusted operational memory |
| Current user issue | Untrusted input |
| Tool result | Untrusted current data |

Critical rules are enforced in code:

- `issue_refund()` checks the refund store and is idempotent.
- The executor requires `check_refund` before `issue_refund`.
- The store starts executions only for an active procedure ID.
- Tool names come from a fixed allowlist.
- Unknown or inconsistent transaction states escalate.
- Customer status and ETA text is built from current tool results.
- User text is never interpreted as procedure administration.
- SQL is parameterized and never exposed to the model.
- No hidden reasoning or chain-of-thought is persisted.

The model receives the exact selected procedure identity, completion and escalation conditions, available tool names, terminal execution state, and current observations. It is instructed to report only grounded facts.

## Local setup

Run commands from this directory:

```bash
cd memory/05-procedural-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in the local `.env`. The committed file contains placeholders and localhost-only development credentials.

Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

Migrations run automatically when the application starts. They are idempotent and create the vector extension, procedure tables, execution tables, constraints, and HNSW index.

Start the CLI:

```bash
python main.py
```

Seed the approved playbooks inside the CLI:

```text
/seed
```

Embedding the seed summaries calls the OpenAI Embeddings API. Re-running `/seed` is idempotent and leaves existing versions unchanged.

## Demo

```text
/procedures
/search Checkout failed but my card was charged
/chat Checkout failed but my card was charged for TXN-DEMO-001
/execution <id shown by chat>
/stats
```

Other deterministic transactions:

```text
/chat I was charged twice for TXN-DUP-002
/chat My refund is delayed for TXN-REFUND-003
/chat Payment state looks wrong for TXN-INCONSISTENT-004
```

The last example escalates because the current transaction/gateway state is unavailable. A request without a transaction ID also escalates after procedure selection.

Admin/demo commands are deliberately separate from `/chat`:

```text
/activate FAILED_PAYMENT_PLAYBOOK 2
/deactivate FAILED_PAYMENT_PLAYBOOK 2
```

In a real service, these operations belong behind authenticated admin APIs and an approval workflow.

## Tests

Tests use a deterministic `FakeEmbeddingProvider` and deterministic support tools. They never require an OpenAI key.

The Docker initialization script creates a separate test database. Run:

```bash
TEST_DATABASE_URL=postgresql://procedural:procedural_dev@localhost:5433/procedural_memory_test \
python -m pytest -q
```

The suite covers persistence, JSONB ordering, semantic retrieval, top-k, category and threshold filters, active-only selection, version transitions, uniqueness, rollback, execution and step state, escalation, immutable definitions, duplicate-refund safety, agent grounding, and the user/admin boundary.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Local API credential; never commit it |
| `OPENAI_MODEL` | Responses API model |
| `OPENAI_EMBEDDING_MODEL` | Embedding model |
| `OPENAI_EMBEDDING_DIMENSIONS` | Must match the migrated vector column |
| `DATABASE_URL` | PostgreSQL application database |
| `TEST_DATABASE_URL` | Separate destructive test database |
| `PROCEDURE_TOP_K` | Number of ranked candidates |
| `PROCEDURE_MAX_TOP_K` | Application search limit |
| `PROCEDURE_MIN_SIMILARITY` | Minimum cosine-derived score |
| `PROCEDURE_MAX_AGENT_TURNS` | Reserved model-loop safety limit |
| `MIGRATIONS_PATH` | SQL migration directory |

## Project structure

```text
memory/05-procedural-memory/
├── main.py
├── agent.py
├── memory.py
├── store.py
├── models.py
├── embeddings.py
├── database.py
├── config.py
├── tools.py
├── procedure_executor.py
├── terminal_ui.py
├── docker-compose.yml
├── migrations/
├── seeds/
└── tests/
```

The repository ignores `.env`, virtual environments, caches, local data directories, and database volume contents. Docker uses a named volume outside the repository.
