"""Offline tests for application-managed and tool-managed agent integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import AgentRunError, DeveloperPreferenceAgent, TOOL_INSTRUCTIONS
from config import Settings
from memory import KVMemoryManager


class FakeResponses:
    def __init__(self, responses) -> None:
        self._responses = iter(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = FakeResponses(responses)


@pytest.fixture
def memory(tmp_path) -> KVMemoryManager:
    manager = KVMemoryManager(tmp_path / "agent.db")
    yield manager
    manager.close()


def settings(tmp_path) -> Settings:
    return Settings(database_path=tmp_path / "agent.db", model="test-model")


@pytest.mark.asyncio
async def test_application_managed_retrieval_injects_only_requested_hits(
    memory: KVMemoryManager, tmp_path
) -> None:
    memory.set("project", "memory-series", "python_version", "3.12")
    memory.set("project", "memory-series", "test_framework", "pytest")
    response = SimpleNamespace(output_text="Use Python 3.12.")
    client = FakeClient([response])
    agent = DeveloperPreferenceAgent(memory, settings(tmp_path), client=client)

    answer = await agent.chat_with_known_memories(
        "What Python version should we use?",
        namespace="project",
        entity_id="memory-series",
        keys=["python_version", "unknown"],
    )

    assert answer == "Use Python 3.12."
    context = client.responses.calls[0]["input"][0]["content"]
    assert '"python_version": "3.12"' in context
    assert "test_framework" not in context
    assert '"unknown"' not in context


@pytest.mark.asyncio
async def test_tool_managed_agent_can_persist_explicit_preference(
    memory: KVMemoryManager, tmp_path
) -> None:
    call = SimpleNamespace(
        type="function_call",
        name="set_memory",
        arguments=json.dumps(
            {
                "namespace": "project",
                "entity_id": "memory-series",
                "key": "test_framework",
                "value_json": '"pytest"',
                "ttl_seconds": None,
            }
        ),
        call_id="call-1",
    )
    first = SimpleNamespace(output=[call], output_text="")
    second = SimpleNamespace(output=[], output_text="Saved pytest as the test framework.")
    client = FakeClient([first, second])
    events = []
    agent = DeveloperPreferenceAgent(
        memory,
        settings(tmp_path),
        client=client,
        on_tool_event=lambda name, arguments, result: events.append((name, arguments, result)),
    )

    answer = await agent.run_tool_agent("For this project, always use pytest for tests.")

    assert answer == "Saved pytest as the test framework."
    assert memory.get("project", "memory-series", "test_framework").value == "pytest"
    assert events[0][0] == "set_memory"
    assert events[0][2]["stored"] is True
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["tool_choice"] == "required"
    assert client.responses.calls[1]["tool_choice"] == "auto"


def test_tool_functions_validate_json_and_ttl(memory: KVMemoryManager, tmp_path) -> None:
    agent = DeveloperPreferenceAgent(memory, settings(tmp_path), client=FakeClient([]))
    invalid_json = agent._execute_tool(
        "set_memory",
        {
            "namespace": "project",
            "entity_id": "memory-series",
            "key": "language",
            "value_json": "not-json",
            "ttl_seconds": None,
        },
    )
    invalid_ttl = agent._execute_tool(
        "set_memory",
        {
            "namespace": "project",
            "entity_id": "memory-series",
            "key": "language",
            "value_json": '"Python"',
            "ttl_seconds": -1,
        },
    )
    assert "error" in invalid_json
    assert "error" in invalid_ttl
    assert memory.list("project", "memory-series") == []


def test_write_policy_is_explicit() -> None:
    assert "explicitly states a durable" in TOOL_INSTRUCTIONS
    assert "Never store greetings" in TOOL_INSTRUCTIONS
    assert "hidden reasoning" in TOOL_INSTRUCTIONS


@pytest.mark.asyncio
async def test_tool_agent_rejects_answer_before_required_lookup(
    memory: KVMemoryManager, tmp_path
) -> None:
    response = SimpleNamespace(output=[], output_text="Use a recent Python version.")
    client = FakeClient([response])
    agent = DeveloperPreferenceAgent(memory, settings(tmp_path), client=client)
    with pytest.raises(AgentRunError, match="required KV lookup"):
        await agent.run_tool_agent("Which Python version should project memory series use?")
    assert client.responses.calls[0]["tool_choice"] == "required"
