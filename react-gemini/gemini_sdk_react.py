"""The same ReAct investigation using Gemini SDK automatic function calling."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import Any

from google import genai
from google.genai import types

import tools as mock_tools
from terminal_ui import (
    ToolEvent,
    missing_evidence,
    show_agent_phase,
    show_continuation,
    show_final,
    show_start,
    show_tool_event,
)

USER_QUERY = "Checkout API latency is high. Find the likely cause."
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_REMOTE_CALLS = 8
MAX_INVESTIGATION_ROUNDS = 3

INSTRUCTIONS = """You are an incident investigation agent.
Investigate using the available tools. Begin with checkout-service and follow degraded
dependencies. Follow observed evidence rather than assumptions.
Use tool results to decide what to inspect next. If evidence contradicts a hypothesis,
change direction. Distinguish observed facts from hypotheses. Do not claim a deeper
root cause unless the evidence supports it. In particular, elevated CPU alone does
not prove CPU saturation or a CPU bottleneck. Stop after identifying the likely
downstream bottleneck; do not keep searching for an internal cause that these tools
cannot establish. In the final answer, state what is observed, what is
a supported inference, and what underlying cause remains unknown. Return only a
concise incident conclusion; do not expose private reasoning.

Before returning a final answer, collect this minimum evidence (in any order):
- checkout-service metrics
- checkout-service dependency health
- payment-service dependency health
- payment-service logs using a query containing "timeout" or "fraud"
- fraud-service metrics
Do not substitute fraud-service logs or deployment history for payment-service logs.
"""

EVENTS: list[ToolEvent] = []


def _require_supported_sdk() -> None:
    installed = version("google-genai")
    try:
        major = int(installed.split(".", 1)[0])
    except ValueError:
        major = 0
    if major < 2:
        raise SystemExit(
            f"google-genai {installed} is installed, but this Chat AFC demo requires "
            "google-genai 2.x. Run: python -m pip install --upgrade -r requirements.txt"
        )


def _final_text(response: Any) -> str:
    """Read final text without triggering the SDK's non-text-parts warning."""
    candidates = getattr(response, "candidates", None) or []
    parts = getattr(candidates[0].content, "parts", None) if candidates else None
    if parts:
        text = "\n".join(part.text for part in parts if getattr(part, "text", None))
        if text:
            return text
        calls = [part.function_call for part in parts if getattr(part, "function_call", None)]
        if calls:
            details = ", ".join(
                f"{call.name}({dict(call.args or {})})" for call in calls
            )
            raise RuntimeError(
                "Gemini returned an unevaluated function call after the SDK AFC "
                f"budget ended: {details}"
            )
    return getattr(response, "text", None) or ""

def _visible_tool_call(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    event = ToolEvent(name, arguments, result)
    EVENTS.append(event)
    show_tool_event(event)


def get_service_metrics(service: str):
    """Get p95 latency, CPU, and database latency for one service.

    Args:
        service: Exact service identifier, such as checkout-service,
            payment-service, or fraud-service.
    """
    result = mock_tools.get_service_metrics(service)
    _visible_tool_call("get_service_metrics", {"service": service}, result)
    return result


def get_recent_deployments(service: str):
    """Get recent deployments for one service.

    Args:
        service: Exact service identifier, such as checkout-service,
            payment-service, or fraud-service.
    """
    result = mock_tools.get_recent_deployments(service)
    _visible_tool_call("get_recent_deployments", {"service": service}, result)
    return result


def get_logs(service: str, query: str):
    """Search logs for one service using a short query.

    Args:
        service: Exact service identifier whose logs should be searched.
        query: Relevant search terms, such as "timeout fraud-service".
    """
    result = mock_tools.get_logs(service, query)
    _visible_tool_call("get_logs", {"service": service, "query": query}, result)
    return result


def get_dependency_health(service: str):
    """Get latency and health for a service's direct dependencies.

    Args:
        service: Exact service identifier whose dependencies should be checked.
    """
    result = mock_tools.get_dependency_health(service)
    _visible_tool_call("get_dependency_health", {"service": service}, result)
    return result


def run() -> None:
    _require_supported_sdk()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running (see README).")

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        temperature=0,
        tools=[
            get_service_metrics,
            get_recent_deployments,
            get_logs,
            get_dependency_health,
        ],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=MAX_REMOTE_CALLS
        ),
    )

    EVENTS.clear()
    show_start(USER_QUERY, "Gemini SDK automatic function-calling loop")
    show_agent_phase()

    with genai.Client(api_key=api_key) as client:
        chat = client.chats.create(model=MODEL, config=config)
        message = USER_QUERY
        final_text = ""
        for round_number in range(1, MAX_INVESTIGATION_ROUNDS + 1):
            response = chat.send_message(message)
            final_text = _final_text(response)
            missing = missing_evidence(EVENTS)
            if not missing:
                break
            if round_number == MAX_INVESTIGATION_ROUNDS:
                break
            show_continuation(missing, round_number + 1)
            missing_list = "\n".join(f"- {item}" for item in missing)
            message = (
                "Your investigation is incomplete. Use the available tools to collect "
                "the missing evidence below before returning a final answer. Do not "
                f"repeat evidence already collected.\n{missing_list}"
            )

    show_final(final_text, EVENTS)


if __name__ == "__main__":
    run()
