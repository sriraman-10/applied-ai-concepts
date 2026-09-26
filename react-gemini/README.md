# ReAct Incident Agent with Gemini

A small learning project that runs the same deterministic production-incident investigation in two ways using the official Gemini Python SDK:

1. `manual_react.py` disables automatic function calling and implements the loop explicitly.
2. `gemini_sdk_react.py` enables automatic function calling so the SDK manages the loop.

All observability tools return hardcoded mock data. They never contact infrastructure, monitoring systems, databases, or other external APIs. Running either agent calls the Gemini API for model decisions.

## ReAct

ReAct combines **reasoning and acting**:

```text
reason / decide -> action -> observation -> repeat -> final answer
```

This demo does not display private chain-of-thought. It prints only observable execution events: tool names, arguments, deterministic observations, and the final answer.

Both programs use the same terminal renderer. Rich tables show each tool action
and observation. After the agent stops, a separate evidence-validation phase
reports `PASS`, `INCOMPLETE`, or an unsupported deeper-cause claim. Validation
does not run missing tools on the agent's behalf.

## Scenario

The user asks:

> Checkout API latency is high. Find the likely cause.

The evidence intentionally forces course correction:

1. Checkout p95 latency is high, while its CPU and database latency are normal.
2. Checkout has no recent deployment.
3. `payment-service` is a degraded checkout dependency.
4. Payment logs show timeouts while calling `fraud-service`.
5. Fraud-service latency is highly degraded.
6. Fraud's local database remains healthy at 16 ms, so the evidence supports
   `fraud-service` as the likely downstream bottleneck, propagating along
   `fraud-service -> payment-service -> checkout-service`. The available data
   does not establish why fraud-service itself is slow.

## Manual loop versus SDK loop

`manual_react.py` makes the orchestration visible:

```text
Gemini response -> parse function call -> execute Python tool
-> append function response -> call Gemini again
```

It uses a bounded `while` loop with a maximum of eight model iterations.

`gemini_sdk_react.py` creates a Gemini chat and passes typed Python functions to
it. `chat.send_message(...)` owns the automatic function-calling loop: the SDK
creates declarations, executes requested functions, returns observations to the
model, and continues until it receives final text. `maximum_remote_calls=8`
bounds that loop. Tool wrappers provide logging without reimplementing the loop.

The application provides the tools, agent instructions, deterministic mock
environment, visible logging, and post-run evidence validation. Gemini controls
which tools to call and in what order. Because function calling uses `AUTO`, it
may finish before gathering enough evidence. In that case validation reports
`INCOMPLETE` and lists what is missing; it does not complete the investigation
silently.

The SDK example uses a bounded completion gate around Chat AFC. If Gemini tries
to finish early, the application sends the missing-evidence list back to the same
chat and asks it to continue, for at most three investigation rounds. Each
`chat.send_message(...)` still delegates function selection, execution, tool
responses, and continuation to SDK automatic function calling. The application
never invokes a missing tool on the agent's behalf. If the evidence contract is
still incomplete after the bound, Phase 2 reports `INCOMPLETE`.

The agent instructions define a minimum evidence contract before completion:
checkout metrics, checkout dependencies, payment dependencies, payment logs that
mention the fraud call, and fraud metrics. The model still chooses the order and
may gather additional evidence. This contract makes successful runs repeatable
without moving tool execution into the validation phase.

The manual version makes this sequence explicit in application code:

```text
model -> tool call -> execution -> observation -> next model call
```

The SDK version delegates that sequence to Gemini Chat automatic function
calling. ReAct requires observable actions and observations; it does not require
printing private chain-of-thought.

## Setup

Use Python 3.10 or newer:

```bash
cd /Users/sriraman10/Documents/blog/applied-ai-concepts/react-gemini
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade -r requirements.txt
cp .env.example .env
```

This demo requires `google-genai` 2.x because Gemini Chat owns the automatic
function-calling loop. Confirm the active interpreter before running:

```bash
python -c "import google.genai; print(google.genai.__version__)"
```

If it prints `1.x`, upgrade with the installation command above. The program also
checks this at startup and exits with a direct instruction instead of returning
an unevaluated function call.

Edit `.env`, then load it:

```bash
set -a
source .env
set +a
```

## Run

Manual ReAct loop:

```bash
python manual_react.py
```

Gemini SDK automatic loop:

```bash
python gemini_sdk_react.py
```

## Test without an API key

```bash
python -m unittest discover -s tests -v
```

The offline tests cover complete and incomplete evidence paths, both entry
points, and rejection of unsupported deeper-cause claims. They make no network
calls.

## References

- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
