# Prompt Caching — OpenAI and Gemini

This fifth standalone project compares OpenAI's **automatic prefix caching** and Gemini's **explicit cache resource** using the same fictional travel-support policy and customer cases. Both demos read actual cached-token counts from API usage.

The idea comes from [Arpit Bhayani's prompt-caching prototype](https://github.com/arpitbbhayani/prototypes-ai/tree/master/prompt-caching), with a new use case and OpenAI-specific behavior. Unlike Gemini's explicit cache resource, OpenAI's supported models automatically cache eligible repeated prefixes. The stable policy goes first; the changing case goes last.

The demo runs three phases:

| Phase | Prompt layout | What it illustrates |
|---|---|---|
| Varied prefix | A distinct first line for each case, then the policy | Matching-prefix opportunities are reduced |
| Stable prefix | Identical policy, then each changing case | Reuse can produce `cached_tokens` |
| Changed prefix | Policy revision changes near the beginning | A revision can reduce reuse of the old prefix |

These are live observations, not a controlled cache-off experiment. OpenAI may cache eligible prefixes automatically, and hit counts can vary with routing, timing, prior traffic, model behavior, and policy length. A cache hit does **not** skip input token accounting: `cached_tokens` is a subset of `input_tokens`. Cache reads may have a lower rate than ordinary input; consult [current OpenAI pricing](https://platform.openai.com/pricing) for the selected model. The script reports measured tokens and latency without hard-coded dollar estimates.

## Run the OpenAI version

```bash
cd prompt-caching
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python openai_demo.py
```

The default model is `gpt-4.1-mini`; override it with `--model` or `OPENAI_MODEL`. The demo makes 12 billable API calls. It uses `store=False` for Responses API response storage; this does not disable prompt caching. Keep the calls close together to improve the chance of reuse. The fictional `policy.txt` can be edited to try revisions.

The example supplies a stable `prompt_cache_key` to improve cache routing for this older model. The policy is deliberately longer than the earlier version: a shared prefix that is too short may never reach an eligible cache breakpoint. The key still cannot force a hit. If the stable phase shows zero cached tokens, check the [OpenAI prompt-cache diagnostics](https://platform.openai.com/docs/guides/prompt-caching) for your account and model; latency differences alone do not demonstrate caching.

## Run the Gemini version

Install the same dependencies, then run:

```bash
export GEMINI_API_KEY="your-key"
python gemini.py
```

The default is `gemini-2.5-flash`; `--model` or `GEMINI_MODEL` overrides it. [Gemini's explicit cache API](https://ai.google.dev/gemini-api/docs/caching) requires a model-specific minimum cache size. The shared policy is intended to clear the 2,048-token minimum documented for Gemini 2.5 Flash; if cache creation rejects it, check the current minimum for your model and expand the useful policy text.

`gemini.py` runs these phases with the **same five questions** as `openai_demo.py`:

| Phase | Gemini call | What is measured |
|---|---|---|
| No explicit cache | Send policy plus each question | Baseline without a named cache; Gemini may still cache implicitly |
| Policy revision A | `client.caches.create(...)`, then pass `cached_content=cache.name` for all five questions | Explicit cache reuse |
| Policy revision B | Create a **new** cache for the changed policy, then use it for the first two questions | A policy edit requires a different cache resource |

The Gemini run makes 12 generation calls plus two cache creations. It sets each cache's TTL to five minutes and attempts to delete both in `finally`, including after an error. Cache creation, storage duration, input, and output can incur charges. The script prints `usage_metadata.cached_content_token_count` for each response; those cached tokens are part of `prompt_token_count`. Compare tokens rather than interpreting a few latency samples as proof of speedup. See [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) for current rates.

Run offline tests with `python -m unittest discover -s tests -v`.

See [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching) and the [Responses API usage fields](https://platform.openai.com/docs/api-reference/responses/object) for OpenAI service behavior and token reporting.
