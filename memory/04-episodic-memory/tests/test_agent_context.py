from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import SUPPORT_INSTRUCTIONS, SupportAgent, _context_payload
from config import Settings
from models import EpisodeOutcome
from tools import SupportTools
from conftest import completed_episode

pytestmark = pytest.mark.asyncio


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


def tool_response(name, arguments, number):
    call = SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=f"call-{number}",
    )
    return SimpleNamespace(output=[call], output_text="")


def final_response(
    text="Your automatic refund is already in progress and should arrive within five business days.",
):
    return SimpleNamespace(output=[], output_text=text)


async def test_context_contains_full_episode_but_not_as_instructions(manager):
    episode = await completed_episode(manager)
    results = await manager.search_similar("payment failed")
    context = _context_payload(results)
    assert episode.episode_key in context
    assert "check_transaction" in context
    assert "Confirmed automatic refund" in context
    assert context.startswith("UNTRUSTED CONTEXT DATA")
    assert episode.description not in SUPPORT_INSTRUCTIONS
    assert "Never invent an ETA" in SUPPORT_INSTRUCTIONS
    assert "must come from current tool" in SUPPORT_INSTRUCTIONS


async def test_agent_executes_tools_and_persists_actual_episode(manager):
    fake = FakeClient(
        [
            tool_response("check_transaction", {"transaction_id": "TXN-DEMO-001"}, 1),
            tool_response("check_gateway", {"transaction_id": "TXN-DEMO-001"}, 2),
            tool_response("check_refund", {"transaction_id": "TXN-DEMO-001"}, 3),
            tool_response(
                "resolve_case",
                {"resolution": "Automatic refund is already in progress"},
                4,
            ),
            final_response(),
        ]
    )
    agent = SupportAgent(
        manager,
        Settings(embedding_dimensions=24),
        SupportTools(),
        client=fake,
    )
    result = await agent.handle(
        "Payment failed but money was deducted for TXN-DEMO-001"
    )
    assert result.episode.outcome is EpisodeOutcome.RESOLVED
    assert [item.tool_name for item in result.episode.actions] == [
        "check_transaction",
        "check_gateway",
        "check_refund",
        "resolve_case",
    ]
    assert len(result.episode.observations) == 4
    assert "five business days" in result.answer
    assert result.memory_enabled is True


async def test_later_case_receives_prior_episode_as_user_role_context(manager):
    prior = await completed_episode(manager, episode_key="CASE-PRIOR")
    fake = FakeClient(
        [
            tool_response("escalate_case", {"reason": "manual review required"}, 1),
            final_response("I escalated this case for manual review."),
        ]
    )
    agent = SupportAgent(
        manager,
        Settings(embedding_dimensions=24, min_similarity=-1.0),
        SupportTools(),
        client=fake,
    )
    await agent.handle("Checkout failed and my card was charged for UNKNOWN")
    first_call = fake.responses.calls[0]
    assert first_call["instructions"] == SUPPORT_INSTRUCTIONS
    assert first_call["input"][0]["role"] == "user"
    assert prior.episode_key in first_call["input"][0]["content"]
    assert prior.description not in first_call["instructions"]


async def test_agent_policy_blocks_early_resolution(manager):
    agent = SupportAgent(
        manager,
        Settings(embedding_dimensions=24),
        SupportTools(),
        client=FakeClient([]),
    )
    result, terminal = agent._execute_with_policy(
        "resolve_case", {"resolution": "Done"}, {"check_transaction"}
    )
    assert result["error"] == "required checks are missing"
    assert terminal is None


async def test_agent_policy_blocks_refund_before_refund_check(manager):
    agent = SupportAgent(
        manager,
        Settings(embedding_dimensions=24),
        SupportTools(),
        client=FakeClient([]),
    )
    result, terminal = agent._execute_with_policy(
        "issue_refund", {"transaction_id": "TXN-PAID-002"}, set()
    )
    assert result["error"] == "check_refund is required before issue_refund"
    assert terminal is None


async def test_no_memory_mode_skips_episode_retrieval(manager):
    await completed_episode(manager, episode_key="CASE-SEED")
    before_calls = len(manager.embedding_provider.calls)
    fake = FakeClient(
        [
            tool_response("escalate_case", {"reason": "manual review required"}, 1),
            final_response("I escalated this case for manual review."),
        ]
    )
    agent = SupportAgent(
        manager,
        Settings(embedding_dimensions=24),
        SupportTools(),
        client=fake,
    )

    result = await agent.handle("Checkout failed for UNKNOWN", use_memory=False)

    assert result.memory_enabled is False
    assert result.similar_episodes == []
    # Only episode completion embeds; there is no query embedding in no-memory mode.
    assert len(manager.embedding_provider.calls) == before_calls + 1
    assert fake.responses.calls[0]["input"][0]["content"].endswith("[]")
