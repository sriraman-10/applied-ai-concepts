"""Measure OpenAI automatic prompt-prefix caching on fictional travel cases."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

POLICY = Path(__file__).with_name("policy.txt").read_text(encoding="utf-8")
CASES = (
    "My flight leaves tomorrow and I need to change it. What should I do?",
    "The airline canceled my flight. Can I get a refund?",
    "The hotel says it is overbooked. What information do you need?",
    "My bag did not arrive. Who handles tracing it?",
    "I see a charge I do not recognize. What details should I send?",
)


@dataclass(frozen=True)
class Result:
    seconds: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int


def summarize(results: list[Result]) -> dict[str, float]:
    total_input = sum(item.input_tokens for item in results)
    total_cached = sum(item.cached_tokens for item in results)
    return {
        "input": total_input,
        "cached": total_cached,
        "uncached": total_input - total_cached,
        "output": sum(item.output_tokens for item in results),
        "hit_rate": total_cached / total_input if total_input else 0.0,
        "mean_seconds": mean(item.seconds for item in results) if results else 0.0,
    }


def call(client, model: str, policy: str, case: str) -> Result:
    start = perf_counter()
    response = client.responses.create(
        model=model,
        instructions=policy,
        input=case,
        prompt_cache_key="northstar-travel-policy-demo",
        max_output_tokens=180,
        store=False,
    )
    elapsed = perf_counter() - start
    usage = response.usage
    if usage is None:
        raise RuntimeError("OpenAI returned no usage data; cannot measure caching")
    details = usage.input_tokens_details
    return Result(elapsed, usage.input_tokens, details.cached_tokens if details else 0, usage.output_tokens)


def run_phase(client, model: str, name: str, policy: str, cases: tuple[str, ...], cold: bool = False) -> list[Result]:
    print(f"\n{name}")
    results = []
    for index, case in enumerate(cases, 1):
        # A changed first line changes the prefix. This is a cold-prefix illustration,
        # not a server-side cache-off switch.
        variant = f"Unique policy edition for case {index}: {case}\n{policy}" if cold else policy
        result = call(client, model, variant, case)
        results.append(result)
        print(f"  {index}: {result.input_tokens} input, {result.cached_tokens} cached, "
              f"{result.output_tokens} output, {result.seconds:.2f}s")
    stats = summarize(results)
    print(f"  cached share {stats['hit_rate']:.1%}; mean latency {stats['mean_seconds']:.2f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY before running this live API demo")
    from openai import OpenAI

    client = OpenAI()
    print("Prompt caching: fictional travel-support policy")
    print("OpenAI caches eligible matching prefixes automatically; each call is billed.")
    cold = run_phase(client, args.model, "Phase 1: varied prefix", POLICY, CASES, cold=True)
    warm = run_phase(client, args.model, "Phase 2: stable prefix", POLICY, CASES)
    changed = POLICY.replace("revision 2026-09-A", "revision 2026-09-B", 1)
    changed_results = run_phase(client, args.model, "Phase 3: changed prefix", changed, CASES[:2])
    print("\nMeasured totals (cached tokens are a subset of input tokens):")
    for label, results in (("varied", cold), ("stable", warm), ("changed", changed_results)):
        stats = summarize(results)
        print(f"  {label:7} input={stats['input']:.0f} cached={stats['cached']:.0f} "
              f"uncached={stats['uncached']:.0f} output={stats['output']:.0f}")
    print("Cache hits are observed, not guaranteed. Compare token counts and latency; "
          "pricing depends on the selected model and current rates.")
    if not any(item.cached_tokens for item in warm):
        print("No stable-phase cache hits observed. Confirm the shared prefix meets this "
              "model's cache minimum, keep requests close together, and consult OpenAI "
              "prompt-cache diagnostics. A stable cache key helps routing but cannot force a hit.")


if __name__ == "__main__":
    main()
