"""The same investigation using the OpenAI Agents SDK runner."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from agents import Agent, ModelSettings, RunHooks, Runner, function_tool

import tools as mock_tools
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

USER_QUERY = "Checkout API latency is high. Find the likely cause."
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_TURNS_PER_ROUND = 8
MAX_INVESTIGATION_ROUNDS = 3

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

_RUN_EVENTS: ContextVar[list[ToolEvent] | None] = ContextVar("run_events", default=None)


def _record(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    events = _RUN_EVENTS.get()
    if events is None:
        return
    event = ToolEvent(name, arguments, result)
    events.append(event)
    show_tool_event(event)


@function_tool
def get_service_metrics(service: str) -> dict[str, Any]:
    """Get p95 latency, CPU, database latency, and status for one service.

    Args:
        service: Exact service identifier, such as checkout-service or fraud-service.
    """
    result = mock_tools.get_service_metrics(service)
    _record("get_service_metrics", {"service": service}, result)
    return result


@function_tool
def get_recent_deployments(service: str) -> dict[str, Any]:
    """Get deployment history for one service during the last 24 hours.

    Args:
        service: Exact service identifier.
    """
    result = mock_tools.get_recent_deployments(service)
    _record("get_recent_deployments", {"service": service}, result)
    return result


@function_tool
def get_logs(service: str, query: str) -> dict[str, Any]:
    """Search logs for a service and relevant terms.

    Args:
        service: Exact service identifier whose logs should be searched.
        query: Relevant terms, such as "timeout fraud-service".
    """
    result = mock_tools.get_logs(service, query)
    _record("get_logs", {"service": service, "query": query}, result)
    return result


@function_tool
def get_dependency_health(service: str) -> dict[str, Any]:
    """Get latency and health for a service's direct dependencies.

    Args:
        service: Exact service identifier whose dependencies should be checked.
    """
    result = mock_tools.get_dependency_health(service)
    _record("get_dependency_health", {"service": service}, result)
    return result


class VisibleRunHooks(RunHooks):
    """Display model boundaries while tool wrappers display observable actions."""

    def __init__(self) -> None:
        self.model_turn = 0

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self.model_turn += 1
        show_iteration(self.model_turn)


def _continuation_prompt(missing: list[str]) -> str:
    items = "\n".join(f"- {item}" for item in missing)
    return (
        "The evidence contract is incomplete. Call the available tools to collect each "
        "missing observation below, then provide a concise conclusion. Do not repeat "
        f"observations already collected.\n{items}"
    )


def build_agent() -> Agent:
    """Build the agent separately so its configuration is easy to test."""
    return Agent(
        name="Incident Investigation Agent",
        model=MODEL,
        instructions=INSTRUCTIONS,
        tools=[
            get_service_metrics,
            get_recent_deployments,
            get_logs,
            get_dependency_health,
        ],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )


def run() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running (see README).")

    events: list[ToolEvent] = []
    token = _RUN_EVENTS.set(events)
    hooks = VisibleRunHooks()
    agent = build_agent()
    run_input: str | list[Any] = USER_QUERY
    latest_answer = ""

    show_start(USER_QUERY, "OpenAI Agents SDK managed loop")
    show_agent_phase()

    try:
        for round_number in range(1, MAX_INVESTIGATION_ROUNDS + 1):
            result = Runner.run_sync(
                agent,
                run_input,
                hooks=hooks,
                max_turns=MAX_TURNS_PER_ROUND,
            )
            latest_answer = str(result.final_output or "")
            missing = missing_evidence(events)
            if not missing:
                break
            if round_number == MAX_INVESTIGATION_ROUNDS:
                break
            show_continuation(missing, round_number)
            run_input = result.to_input_list()
            run_input.append({"role": "user", "content": _continuation_prompt(missing)})
    finally:
        _RUN_EVENTS.reset(token)

    show_final(latest_answer, events)


if __name__ == "__main__":
    run()
