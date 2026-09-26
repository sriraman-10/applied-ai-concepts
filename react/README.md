# ReAct Incident Investigation with OpenAI

This project runs one deterministic incident scenario in two ways:

- `manual_react.py` implements the function-calling loop directly with the OpenAI Responses API.
- `agents_sdk.py` gives the same tools to the OpenAI Agents SDK, whose runner manages model calls and tool execution.

The comparison is about orchestration. Both programs use the same prompt, mock tools, evidence contract, terminal renderer, and verdict logic.

## What the demo shows

ReAct is a loop in which a model selects an action, observes its result, and decides what to do next:

```text
model decision → tool call → observation → repeat → final answer
```

The terminal displays only observable events. It does not print private chain-of-thought.

The scenario starts with slow checkout requests. The useful evidence follows this path:

```text
checkout-service → payment-service → fraud-service
```

The tools establish that `fraud-service` is the likely downstream bottleneck. They do not establish the internal reason why that service is slow, so the verdict deliberately stops there.

## Reliability boundary

Model tool selection can vary between runs. The application therefore enforces a minimum-evidence contract before accepting a conclusion:

1. degraded checkout metrics
2. checkout deployment history
3. checkout dependency health showing degraded payment-service
4. payment dependency health showing degraded fraud-service
5. payment logs showing a fraud-service timeout
6. degraded fraud-service metrics

If the model answers early, the application sends a bounded continuation request listing only the missing observations. It never calls missing tools on the model's behalf. If the evidence remains incomplete, the verdict is `INCONCLUSIVE`.

The model response is displayed as a **Model draft**. The final **Verdict** is derived from the tool results collected during that run. This prevents polished model wording from being presented as a verified root cause.

## Project layout

```text
react/
├── manual_react.py   # manual Responses API loop
├── agents_sdk.py     # Agents SDK runner
├── terminal_ui.py    # Rich output and evidence validation
├── tools.py          # deterministic mock observability data
├── tests/            # offline tests
└── requirements.txt
```

## Setup

Use a project-local virtual environment. This keeps the OpenAI and Agents SDK versions isolated from unrelated projects.

```bash
cd /Users/sriraman10/Documents/blog/applied-ai-concepts/react
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your shell. The scripts do not automatically load `.env`:

```bash
export OPENAI_API_KEY="your-key"
```

The default model is `gpt-5-mini`. To use another model available to your account:

```bash
export OPENAI_MODEL="your-model"
```

## Run

```bash
python manual_react.py
python agents_sdk.py
```

Both commands print:

- a colored investigation header
- model-turn separators
- a table for each tool action and observation
- an evidence summary table
- validation status
- the model draft
- a supported or inconclusive verdict

## Test without API calls

```bash
python -m unittest discover -s tests -v
```

Tests exercise the deterministic tools and evidence validator. They do not require an API key or make network requests.

## Why use the OpenAI Agents SDK here?

The SDK runner has the same control boundary this example is intended to teach: it calls the model, executes requested tools, returns tool results to the model, and stops at final output. Lifecycle hooks expose model-turn boundaries, while the tool wrappers expose the actual calls and results. The application still owns its production policy: evidence requirements, bounded retries, and the final verified verdict.

## References

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Running agents](https://openai.github.io/openai-agents-python/running_agents/)
