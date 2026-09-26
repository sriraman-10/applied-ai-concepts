"""Ralph = fresh Observe -> Think -> Act cycles around persistent files."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
MAX_ITERATIONS = 10
VERIFY_TIMEOUT = 10

# A JSON schema keeps the model's decision separate from its generated code.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["IMPROVE", "COMPLETE"]},
        **{name: {"type": "string"} for name in (
            "approach_name", "explanation", "time_complexity", "space_complexity", "code"
        )},
    },
    "required": ["status", "approach_name", "explanation",
                 "time_complexity", "space_complexity", "code"],
}


def observe() -> str:
    """Re-read all persistent memory on EVERY iteration."""
    files = [WORKSPACE / "problem.md", WORKSPACE / "progress.md"]
    solutions = sorted((WORKSPACE / "solutions").glob("*.py"))
    print("Existing solutions:", ", ".join(p.name for p in solutions) or "None")
    return json.dumps({str(p.relative_to(WORKSPACE)): p.read_text(encoding="utf-8")
                       for p in files + solutions}, indent=2)


def think_with_gemini(client, state: str) -> dict:
    """A stateless generate_content call: no chat/session or past messages."""
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        contents="Current filesystem snapshot:\n" + state,
        config={
            "system_instruction": (ROOT / "prompt.md").read_text(encoding="utf-8"),
            "response_mime_type": "application/json",
            "response_json_schema": DECISION_SCHEMA,
        },
    )
    decision = json.loads(response.text or "{}")
    if not isinstance(decision, dict) or any(
        not isinstance(decision.get(key), str) for key in DECISION_SCHEMA["required"]
    ):
        raise ValueError("Missing or invalid decision fields")
    if decision["status"] not in ("IMPROVE", "COMPLETE"):
        raise ValueError("Unknown decision status")
    if decision["status"] == "IMPROVE" and not decision["code"].strip():
        raise ValueError("IMPROVE requires code")
    return decision


def act(decision: dict) -> Path:
    """Persist a new version; never overwrite earlier attempts."""
    folder = WORKSPACE / "solutions"
    versions = [int(m.group(1)) for p in folder.iterdir()
                if (m := re.fullmatch(r"solution_v(\d+)\.py", p.name))]
    path = folder / f"solution_v{max(versions, default=0) + 1}.py"
    header = "\n".join(
        "# " + line for key in ("approach_name", "time_complexity", "space_complexity")
        for line in f"{key}: {decision[key]}".splitlines()
    )
    with path.open("x", encoding="utf-8") as file:
        file.write(header + "\n\n" + decision["code"] + "\n")
    print("Created:", path.relative_to(ROOT))
    return path


def verify(path: Path) -> tuple[bool, str]:
    """A separate process bounds hangs; it is NOT a security sandbox."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    with tempfile.TemporaryDirectory() as directory:
        # File-backed output avoids accumulating unbounded stdout in RAM.
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(
                    [sys.executable, "-I", str(ROOT / "tests" / "test_solutions.py"),
                     str(path.resolve())],
                    cwd=directory, env=env, stdout=output, stderr=output,
                    timeout=VERIFY_TIMEOUT, check=False,
                )
            except subprocess.TimeoutExpired:
                return False, f"FAIL: verification exceeded {VERIFY_TIMEOUT}s"
            output.seek(0)
            details = output.read(8000).decode("utf-8", errors="replace")
    passed = result.returncode == 0
    return passed, ("PASS" if passed else "FAIL") + "\n" + details


def persist(entry: str) -> None:
    with (WORKSPACE / "progress.md").open("a", encoding="utf-8") as file:
        file.write("\n" + entry + "\n")


def run(client) -> bool:
    persist("## New run\nSTATUS: RUNNING")
    # The OUTER Ralph loop. No model conversation is carried around this loop.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 50}\nRALPH ITERATION {iteration}\n{'=' * 50}")
        print("Observing workspace...")
        state = observe()                              # OBSERVE
        print("Asking Gemini for the next improvement...")
        try:
            decision = think_with_gemini(client, state)  # THINK (fresh request)
        except (ValueError, TypeError) as error:
            persist(f"### Iteration {iteration}\nInvalid response: {type(error).__name__}")
            print("Invalid structured response; retrying in the next iteration.")
            continue
        print("Gemini proposed:", decision["approach_name"])
        entry = (f"### Iteration {iteration}\nDecision: {decision['status']}\n"
                 f"Approach: {decision['approach_name']}\n"
                 f"Time: {decision['time_complexity']}\n"
                 f"Space: {decision['space_complexity']}\n"
                 f"Explanation: {decision['explanation']}\n")

        if decision["status"] == "COMPLETE":
            # Recheck persisted artifacts, including on a resumed run.
            checks = [(p, verify(p)) for p in sorted(
                (WORKSPACE / "solutions").glob("*.py"))]
            for path, (_, details) in checks:
                entry += f"\nRechecked {path.name}: {details}\n"
            if any(passed for _, (passed, _) in checks):
                persist(entry + "\nSTATUS: COMPLETE")
                print("STATUS: COMPLETE")
                return True
            persist(entry + "\nCompletion rejected: no solution passes verification.")
            print("Completion rejected: no verified solution. Continuing...")
            continue

        path = act(decision)                           # ACT + PERSIST code
        _, details = verify(path)                      # VERIFY independently
        persist(entry + f"File: {path.name}\nVerification: {details}")
        print("Verification:", details)
        print("Continuing Ralph loop...")              # REPEAT, even on failure

    persist("STATUS: MAX_ITERATIONS_REACHED")
    print("STATUS: MAX_ITERATIONS_REACHED")
    return False



def describe_error(error: Exception) -> str:
    """Show the API reason without dumping request bodies or credentials."""
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None) or str(error)
    summary = f"{type(error).__name__} ({code or 'unknown'} {status or ''}): {message}"
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            summary = summary.replace(key, "[REDACTED]")
    summary = re.sub(r"AIza[\w-]+", "[REDACTED]", summary)
    hints = {
        400: "Check the API message for an invalid key or unsupported request configuration.",
        401: "Check that GEMINI_API_KEY is valid and exported in this shell.",
        403: "Check the key's API restrictions and project access to the Gemini API.",
        404: "Check GEMINI_MODEL: the selected model may be unavailable to your account.",
        429: "Check Gemini API quota and billing; retry after the indicated delay.",
    }
    return summary[:3000] + "\n" + hints.get(code, "Check API access and configuration.")


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key == "replace_with_your_key":
        print("Set GEMINI_API_KEY before running (see README).", file=sys.stderr)
        return 1
    from google import genai  # Offline tests do not need the SDK or a key.

    try:
        with genai.Client(api_key=key) as client:
            return 0 if run(client) else 1
    except Exception as error:
        # Keep API diagnostics out of the persisted prompt; show a redacted reason.
        persist(f"STATUS: ERROR\nRun interrupted: {type(error).__name__}")
        print("Run interrupted: " + describe_error(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
