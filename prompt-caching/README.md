# Prompt Caching — OpenAI Responses API

This fifth standalone project demonstrates OpenAI's **automatic prefix caching** using a fictional travel-support policy. Each request applies the same long policy to a different customer case. The client reads the API's `cached_tokens` field rather than assuming that a repeated prompt was cached.

The idea comes from [Arpit Bhayani's prompt-caching prototype](https://github.com/arpitbbhayani/prototypes-ai/tree/master/prompt-caching), with a new use case and OpenAI-specific behavior. Unlike Gemini's explicit cache resource, OpenAI's supported models automatically cache eligible repeated prefixes. The stable policy goes first; the changing case goes last.

The demo runs three phases:

| Phase | Prompt layout | What it illustrates |
|---|---|---|
| Varied prefix | A distinct first line for each case, then the policy | Matching-prefix opportunities are reduced |
| Stable prefix | Identical policy, then each changing case | Reuse can produce `cached_tokens` |
| Changed prefix | Policy revision changes near the beginning | A revision can reduce reuse of the old prefix |

These are live observations, not a controlled cache-off experiment. OpenAI may cache eligible prefixes automatically, and hit counts can vary with routing, timing, prior traffic, model behavior, and policy length. A cache hit does **not** skip input token accounting: `cached_tokens` is a subset of `input_tokens`. Cache reads may have a lower rate than ordinary input; consult [current OpenAI pricing](https://platform.openai.com/pricing) for the selected model. The script reports measured tokens and latency without hard-coded dollar estimates.

## Run

```bash
cd prompt-caching
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python main.py
```

The default model is `gpt-4.1-mini`; override it with `--model` or `OPENAI_MODEL`. The demo makes 12 billable API calls. It uses `store=False` for Responses API response storage; this does not disable prompt caching. Keep the calls close together to improve the chance of reuse. The fictional `policy.txt` can be edited to try revisions.

The example supplies a stable `prompt_cache_key` to improve cache routing for this older model. The policy is deliberately longer than the earlier version: a shared prefix that is too short may never reach an eligible cache breakpoint. The key still cannot force a hit. If the stable phase shows zero cached tokens, check the [OpenAI prompt-cache diagnostics](https://platform.openai.com/docs/guides/prompt-caching) for your account and model; latency differences alone do not demonstrate caching.

Run offline tests with `python -m unittest discover -s tests -v`.

See [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching) and the [Responses API usage fields](https://platform.openai.com/docs/api-reference/responses/object) for service behavior and token reporting.
