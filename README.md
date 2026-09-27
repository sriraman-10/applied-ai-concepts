# Applied AI Concepts

Small, runnable projects that explain AI-agent architecture by making the important mechanics visible in code.

Each project isolates one concept—agent loops, tool use, evidence validation, or memory—and implements it without hiding the core behavior behind a large framework. The examples favor deterministic fixtures, observable terminal output, explicit safety boundaries, and offline tests.

## Projects

| Project | Concept | Runtime |
|---|---|---|
| [Ralph Loop](ralph-loop/) | An outer loop that observes persistent state, asks a fresh worker for one improvement, acts, verifies, and repeats | Gemini SDK |
| [ReAct with OpenAI](react/) | Manual Responses API function calling compared with an Agents SDK-managed loop | OpenAI Responses API and Agents SDK |
| [ReAct with Gemini](react-gemini/) | Manual function calling compared with Gemini SDK automatic function calling | Gemini SDK |
| [Memory #1 — Working Memory](memory/01-working-memory/) | Token budgeting, noise eviction, summary compaction, recent-context protection, and hard-eviction safety | OpenAI Responses API |
| [Memory #2 — KV Memory](memory/02-kv-memory/) | Persistent exact-key memory, SQLite lifecycle management, and application-managed versus tool-managed retrieval | SQLite and OpenAI Responses API |
| [Memory #3 — Vector Memory](memory/03-vector-memory/) | Persistent semantic memory, embedding-based retrieval, ChromaDB lifecycle management, and untrusted-data boundaries | ChromaDB, OpenAI Embeddings, and Responses API |
| [Memory #4 — Episodic Memory](memory/04-episodic-memory/) | Complete support-case experiences, structured lifecycle capture, and similar-case retrieval | PostgreSQL, pgvector, OpenAI Embeddings, and Responses API |

## Learning map

```text
Agent execution
├── Ralph loop
│   └── repeat work across fresh model calls and persistent files
└── ReAct
    └── model → tool → observation → repeat → answer

Agent memory
├── Working memory
│   └── bounded context, token pressure, eviction, and compaction
├── Key-value memory
│   └── persistent structured facts retrieved by exact key
├── Vector memory
│   └── persistent knowledge retrieved by semantic similarity
└── Episodic memory
    └── complete past experiences retrieved as similar-case evidence
```

The memory series intentionally separates memory types. Working memory manages what fits into the current model context. KV memory stores durable facts outside the model and retrieves them using a known namespace, entity, and key. Vector memory retrieves durable knowledge by semantic similarity. Episodic memory stores complete experiences with their actions, observations, resolutions, and outcomes. Later projects can add procedural memory without combining these different retrieval and lifecycle rules.

## Repository principles

- **Make orchestration visible.** Manual implementations show the model/tool boundary before introducing SDK-managed alternatives.
- **Keep model output separate from verified facts.** Incident demos derive their verdict from observed tool evidence.
- **Own memory policy explicitly.** Storage, retrieval, compaction, expiration, and deletion remain application concerns.
- **Use deterministic local fixtures.** Demonstrations do not connect to production infrastructure.
- **Keep private reasoning private.** The terminal shows actions, arguments, observations, summaries, and answers—not hidden chain-of-thought.
- **Test without API calls.** Core behavior uses fake clients or deterministic data so tests do not require credentials.

## Getting started

Clone the repository and enter the project you want to run:

```bash
git clone https://github.com/sriraman-10/applied-ai-concepts.git
cd applied-ai-concepts
cd memory/02-kv-memory  # example
```

Each project has its own README, dependencies, setup commands, environment variables, and test instructions. Use a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the project’s `.env.example` only when the demo needs API access:

```bash
cp .env.example .env
```

Replace placeholder values locally. Real `.env` files, virtual environments, generated caches, and runtime databases are ignored by Git.

## Testing

Run tests from the individual project directory. Examples:

```bash
# Working-memory project
cd memory/01-working-memory
python -m pytest -q

# OpenAI ReAct project
cd ../../react
python -m unittest discover -s tests -v
```

The detailed project README explains what each suite verifies and whether running the demo makes external API calls.

## Repository structure

```text
applied-ai-concepts/
├── ralph-loop/
├── react/
├── react-gemini/
└── memory/
    ├── 01-working-memory/
    ├── 02-kv-memory/
    ├── 03-vector-memory/
    └── 04-episodic-memory/
```

Start with the README inside a project, run its tests, and then run the demo to observe the concept in the terminal.
