"""A visible ReAct loop using Gemini function calling manually."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from tools import execute_tool
from terminal_ui import (
    ToolEvent,
    show_agent_phase,
    show_final,
    show_iteration,
    show_start,
    show_tool_event,
)

USER_QUERY = "Checkout API latency is high. Find the likely cause."
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_ITERATIONS = 8

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

FUNCTION_DECLARATIONS = [
    {
        "name": "get_service_metrics",
        "description": "Get p95 latency, CPU, and database latency for one service.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string", "description": "Service name"}},
            "required": ["service"],
        },
    },
    {
        "name": "get_recent_deployments",
        "description": "Get recent deployments for one service.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string", "description": "Service name"}},
            "required": ["service"],
        },
    },
    {
        "name": "get_logs",
        "description": "Search logs for one service using a short query.",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service name"},
                "query": {"type": "string", "description": "Terms to find in the logs"},
            },
            "required": ["service", "query"],
        },
    },
    {
        "name": "get_dependency_health",
        "description": "Get latency and health for a service's direct dependencies.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string", "description": "Service name"}},
            "required": ["service"],
        },
    },
]


def run() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running (see README).")

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=USER_QUERY)])
    ]
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        temperature=0,
        tools=[types.Tool(function_declarations=FUNCTION_DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    events: list[ToolEvent] = []
    show_start(USER_QUERY, "Manual function-calling loop")
    show_agent_phase()
    iteration = 1

    with genai.Client(api_key=api_key) as client:
        while iteration <= MAX_ITERATIONS:
            show_iteration(iteration)
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )

            model_content = response.candidates[0].content
            contents.append(model_content)
            tool_calls = response.function_calls or []

            if not tool_calls:
                final_text = response.text or ""
                if final_text.strip():
                    show_final(final_text, events)
                    return
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text="Continue the investigation or provide the final answer.")],
                    )
                )
                iteration += 1
                continue

            function_responses: list[types.Part] = []
            for tool_call in tool_calls:
                arguments = dict(tool_call.args or {})
                observation = execute_tool(tool_call.name or "", arguments)
                event = ToolEvent(tool_call.name or "", arguments, observation)
                events.append(event)
                show_tool_event(event)

                function_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=tool_call.id,
                            name=tool_call.name,
                            response={"result": observation},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=function_responses))
            iteration += 1

    raise RuntimeError(f"Agent did not finish within {MAX_ITERATIONS} iterations.")


if __name__ == "__main__":
    run()
