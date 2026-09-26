"""A visible ReAct loop using the OpenAI Responses API directly."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from terminal_ui import (
    ToolEvent,
    missing_evidence,
    show_agent_phase,
    show_continuation,
    show_final,
    show_iteration,
    show_start,
    show_tool_event,
)
from tools import call_tool

USER_QUERY = "Checkout API latency is high. Find the likely cause."
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_MODEL_TURNS = 10

INSTRUCTIONS = """You are an incident investigation agent using deterministic mock tools.
Begin with checkout-service and follow degraded dependencies. Use observed evidence rather
than assumptions. If evidence contradicts a hypothesis, change direction. Distinguish
observed facts from hypotheses. Elevated CPU alone does not prove CPU saturation.
Do not claim why the final downstream service is slow unless a tool establishes the cause.
Return only a concise incident conclusion; never expose private chain-of-thought.

Before returning a final answer, collect this minimum evidence in any order:
- degraded checkout-service metrics
- checkout-service deployment history
- checkout-service dependency health showing payment-service
- payment-service dependency health showing fraud-service
- payment-service logs using a query containing "timeout" or "fraud"
- degraded fraud-service metrics
Do not substitute fraud-service logs for payment-service logs.
"""


def function_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Create a strict Responses API function-tool schema."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    function_tool(
        "get_service_metrics",
        "Get p95 latency, CPU, database latency, and status for one service.",
        {"service": {"type": "string", "description": "Exact service identifier"}},
    ),
    function_tool(
        "get_recent_deployments",
        "Get deployment history for one service during the last 24 hours.",
        {"service": {"type": "string", "description": "Exact service identifier"}},
    ),
    function_tool(
        "get_logs",
        "Search one service's logs. For the dependency path, search payment-service for timeout or fraud-service.",
        {
            "service": {"type": "string", "description": "Exact service identifier"},
            "query": {"type": "string", "description": "Relevant terms such as 'timeout fraud-service'"},
        },
    ),
    function_tool(
        "get_dependency_health",
        "Get latency and health for a service's direct dependencies.",
        {"service": {"type": "string", "description": "Exact service identifier"}},
    ),
]


def _continuation_prompt(missing: list[str]) -> str:
    items = "\n".join(f"- {item}" for item in missing)
    return (
        "The evidence contract is incomplete. Call the available tools to collect each "
        "missing observation below, then provide a concise conclusion. Do not repeat "
        f"observations already collected.\n{items}"
    )


def run() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running (see README).")

    client = OpenAI()
    conversation: list[Any] = [{"role": "user", "content": USER_QUERY}]
    events: list[ToolEvent] = []
    latest_answer = ""
    continuation = 0

    show_start(USER_QUERY, "Manual OpenAI Responses API loop")
    show_agent_phase()

    for turn in range(1, MAX_MODEL_TURNS + 1):
        show_iteration(turn)
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            input=conversation,
            parallel_tool_calls=False,
        )
        conversation.extend(response.output)
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.arguments)
                except (TypeError, json.JSONDecodeError) as error:
                    result = {"error": f"Invalid JSON arguments: {error}"}
                    arguments = {}
                else:
                    result = call_tool(tool_call.name, arguments)

                event = ToolEvent(tool_call.name, arguments, result)
                events.append(event)
                show_tool_event(event)
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(result, sort_keys=True),
                    }
                )
            continue

        latest_answer = response.output_text or ""
        missing = missing_evidence(events)
        if not missing:
            break
        continuation += 1
        show_continuation(missing, continuation)
        conversation.append({"role": "user", "content": _continuation_prompt(missing)})

    show_final(latest_answer, events)


if __name__ == "__main__":
    run()
