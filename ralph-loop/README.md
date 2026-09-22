# Ralph Loop — From First Principles

Ralph is an outer loop that repeatedly invokes a fresh worker against persistent
state until a goal is complete. Here that worker is one Gemini request, and its
task is to discover and improve a Longest Increasing Subsequence (LIS) algorithm.
Python controls the loop; Gemini chooses the approach.

**OTA = loop/work cycle inside an agent iteration.** Observe reads the workspace,
Think asks Gemini for one decision, and Act writes a new solution.
**Ralph = loop around repeated fresh agent/model invocations.** Verification and
persistence close each iteration before the next one starts.

```text
                    RALPH LOOP
               ┌─────────────────┐
               │ OBSERVE         │
               │ problem.md      │
               │ progress.md     │
               │ solutions/*.py  │
               └────────┬────────┘
                        ↓
               ┌─────────────────┐
               │ THINK           │
               │ Fresh Gemini    │
               │ request         │
               └────────┬────────┘
                        ↓
               ┌─────────────────┐
               │ ACT             │
               │ Create a new    │
               │ solution file   │
               └────────┬────────┘
                        ↓
               ┌─────────────────┐
               │ VERIFY          │
               │ Deterministic   │
               │ tests           │
               └────────┬────────┘
                        ↓
                  Persist progress
                        ↓
                  Goal complete?
                    /       \
                  No         Yes
                  │           │
                  └── loop    STOP
```

A COMPLETE decision takes the completion branch without creating a file; existing
solutions are rechecked before accepting it. With no passing solution, the loop
records the rejection and continues.

The filesystem is the memory between iterations. Every request receives the
problem, progress, and **all** solution modules freshly read from disk. There is
no chat session or accumulated sequence of model messages. Reusing the SDK client
only reuses the connection; it does not carry conversational memory. A restart
reads the same files and continues version numbering without overwriting attempts.
The last STATUS line in the append-only progress log describes the current run.

An illustrative run might move from exhaustive search, to quadratic dynamic
programming, to an O(n log n) approach, then COMPLETE. This sequence is neither
hardcoded nor guaranteed: Gemini may jump directly to a strong solution. The
prompt asks for a simple starting point and meaningful improvements, without
prescribing algorithms or iteration numbers.

Finite tests check behavior, not asymptotic complexity or mathematical optimality.
The complexity labels and stopping justification are model judgments to inspect.
COMPLETE means Gemini sees no meaningful further improvement and at least one
persisted solution passes tests. MAX_ITERATIONS (10 per invocation) bounds cost
and retries when the model repeats itself or cannot finish. Exhausting it records
STATUS: MAX_ITERATIONS_REACHED and exits with a nonzero status.

## Run

Use Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey).
From the repository root, on macOS/Linux:

If the checkout already contains generated solutions, follow **Start a fresh run**
below before running the demo to see it build solutions from scratch.

```bash
cd ralph-loop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and replace replace_with_your_key with your API key.
set -a
source .env
set +a
python ralph.py
```

`.env` is loaded by the shell, not automatically by Python; it is gitignored.
Alternatively, with the virtual environment active:

```bash
GEMINI_API_KEY='your-key' python ralph.py
```

`GEMINI_MODEL` defaults to `gemini-2.5-flash`; set it to a Gemini model available
to your account that supports structured output. Requests can incur API charges.
API/configuration errors stop the run and record STATUS: ERROR. To resume, fix
the issue and run the same command. The terminal shows the API status and reason
with API keys redacted; progress.md records the error type.

## Existing solutions and reruns

Running `python3 ralph.py` again resumes from the files already in `workspace/`.
It still makes a fresh Gemini request. If an optimized, passing solution already
exists, Gemini may return `COMPLETE` in the first iteration without creating any
new solution. Existing files do not prevent execution; they can make the goal
already complete. If improvement or repair is needed, the loop creates a new
version and preserves earlier attempts.

To see the full progression again, reset **both** the generated solutions and
progress.md. Deleting only solutions leaves the previous run's history in the
next request.

## Start a fresh run

From the `ralph-loop` directory, with the virtual environment active and your
API key exported, run the following. This deletes generated Python solutions and
clears the run history; save them elsewhere first if you want to keep them.
The problem, prompt, API-key file, and `.gitkeep` remain in place.

```bash
python3 - <<'PYTHON'
from pathlib import Path

workspace = Path("workspace")
for solution in (workspace / "solutions").glob("*.py"):
    solution.unlink()
(workspace / "progress.md").write_text(
    "# Ralph progress\n\n"
    "STATUS: NOT_STARTED\n\n"
    "No solutions yet. Each iteration appends its decision and verification result.\n"
    "The last STATUS line is the current run status.\n",
    encoding="utf-8",
)
PYTHON
python3 ralph.py
```

## Example successful run

This actual run started from an empty solution directory. Gemini selected dynamic
programming, improved it with patience sorting and binary search, then declared
completion. Both generated solutions passed all 131 deterministic cases.
Other runs may choose different approaches or use fewer or more iterations.

```text
$ python3 ralph.py

==================================================
RALPH ITERATION 1
==================================================
Observing workspace...
Existing solutions: None
Asking Gemini for the next improvement...
Gemini proposed: Dynamic Programming
Created: workspace/solutions/solution_v1.py
Verification: PASS
PASS: 131 deterministic cases

Continuing Ralph loop...

==================================================
RALPH ITERATION 2
==================================================
Observing workspace...
Existing solutions: solution_v1.py
Asking Gemini for the next improvement...
Gemini proposed: Patience Sorting with Binary Search
Created: workspace/solutions/solution_v2.py
Verification: PASS
PASS: 131 deterministic cases

Continuing Ralph loop...

==================================================
RALPH ITERATION 3
==================================================
Observing workspace...
Existing solutions: solution_v1.py, solution_v2.py
Asking Gemini for the next improvement...
Gemini proposed: Patience Sorting with Binary Search
STATUS: COMPLETE
```

The third iteration rechecks existing solutions and records completion in
`workspace/progress.md`; it does not create `solution_v3.py`.

## Verify without an API key

```bash
python -m unittest discover -s tests -v
# Check one generated solution directly:
python tests/test_solutions.py workspace/solutions/solution_v1.py
```

Tests use only the standard library. Offline integration tests script model
responses to exercise failure repair, fresh filesystem snapshots, completion
rejection, persistence across runs, malformed responses, and verification timeouts.
The generated-solution test skips until files exist. Afterwards it checks every
attempt, so historical failed attempts intentionally remain visible as failures.

Each solution must expose `length_of_lis(nums: list[int]) -> int`. The verifier
checks empty inputs, duplicates, negatives, mixed sequences, and exhaustive small
inputs against an independent oracle (131 cases total). Failed code and diagnostic
output remain on disk so the next invocation can repair the problem. A ten-second
subprocess timeout catches hangs. Generated Python executes locally: this process
boundary is not a security sandbox. Use a disposable environment for isolation.

## Read the implementation

- `ralph.py`: the visible outer loop and small observe/think/act/verify functions.
- `prompt.md`: the worker instructions; no prescribed algorithm sequence.
- `workspace/`: persistent problem, append-only progress, immutable solution versions.
- `tests/`: deterministic solution checks and offline loop tests.

This intentionally uses Python and the official `google-genai` SDK, without an
agent framework, so the mechanics remain visible. See the
[official SDK documentation](https://googleapis.github.io/python-genai/) for
`generate_content` and JSON-schema structured output.
