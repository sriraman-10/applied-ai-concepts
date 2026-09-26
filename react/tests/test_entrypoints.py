"""Offline integration tests for both orchestration entrypoints."""

from __future__ import annotations

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rich.console import Console

import agents_sdk
import manual_react
import terminal_ui
from terminal_ui import ToolEvent
from tools import call_tool


CALLS = [
    ("get_service_metrics", {"service": "checkout-service"}),
    ("get_recent_deployments", {"service": "checkout-service"}),
    ("get_dependency_health", {"service": "checkout-service"}),
    ("get_dependency_health", {"service": "payment-service"}),
    ("get_logs", {"service": "payment-service", "query": "timeout fraud-service"}),
    ("get_service_metrics", {"service": "fraud-service"}),
]

FINAL = (
    "fraud-service is the likely downstream bottleneck. Its latency propagates "
    "through payment-service to checkout-service. Its internal cause is unknown."
)


class FakeManualResponse:
    def __init__(self, output, text="") -> None:
        self.output = output
        self.output_text = text


class FakeRunResult:
    def __init__(self, final_output: str, history=None) -> None:
        self.final_output = final_output
        self._history = list(history or [])

    def to_input_list(self):
        return list(self._history)


class EntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.console_patch = patch.object(
            terminal_ui,
            "console",
            Console(file=self.output, force_terminal=False, width=120),
        )
        self.console_patch.start()
        self.addCleanup(self.console_patch.stop)

    def assert_supported_output(self) -> None:
        rendered = self.output.getvalue()
        self.assertIn("Phase 1 · Agent investigation", rendered)
        self.assertIn("Phase 2 · Evidence validation", rendered)
        self.assertIn("Evidence collected by the agent", rendered)
        self.assertIn("PASS", rendered)
        self.assertIn("Model draft", rendered)
        self.assertIn("SUPPORTED", rendered)
        self.assertIn("does not establish why fraud-service itself is slow", rendered)

    def test_manual_loop_collects_and_validates_evidence(self) -> None:
        tool_calls = [
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=__import__("json").dumps(arguments),
                call_id=str(index),
            )
            for index, (name, arguments) in enumerate(CALLS, start=1)
        ]
        responses = iter(
            [FakeManualResponse(tool_calls), FakeManualResponse([], FINAL)]
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: next(responses))
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch.object(
            manual_react, "OpenAI", return_value=client
        ):
            manual_react.run()

        self.assert_supported_output()

    def test_manual_loop_requests_missing_evidence(self) -> None:
        final_calls = [
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=__import__("json").dumps(arguments),
                call_id=str(index),
            )
            for index, (name, arguments) in enumerate(CALLS, start=1)
        ]
        responses = iter(
            [
                FakeManualResponse([], "The cause is downstream."),
                FakeManualResponse(final_calls),
                FakeManualResponse([], FINAL),
            ]
        )
        captured_inputs = []

        def create(**kwargs):
            captured_inputs.append(kwargs["input"])
            return next(responses)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch.object(
            manual_react, "OpenAI", return_value=client
        ):
            manual_react.run()

        self.assertIn("Evidence gate", self.output.getvalue())
        self.assertTrue(
            any("evidence contract is incomplete" in str(item).lower() for item in captured_inputs[1])
        )
        self.assert_supported_output()

    def test_agents_sdk_run_collects_and_validates_evidence(self) -> None:
        def run_sync(*args, **kwargs):
            for name, arguments in CALLS:
                agents_sdk._record(name, arguments, call_tool(name, arguments))
            return FakeRunResult(FINAL)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch.object(
            agents_sdk.Runner, "run_sync", side_effect=run_sync
        ):
            agents_sdk.run()

        self.assert_supported_output()

    def test_agents_sdk_continues_without_backfilling_tools(self) -> None:
        calls = 0
        received_inputs = []

        def run_sync(agent, run_input, **kwargs):
            nonlocal calls
            calls += 1
            received_inputs.append(run_input)
            if calls == 1:
                return FakeRunResult("The cause is downstream.", [{"role": "user", "content": "initial"}])
            for name, arguments in CALLS:
                agents_sdk._record(name, arguments, call_tool(name, arguments))
            return FakeRunResult(FINAL)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch.object(
            agents_sdk.Runner, "run_sync", side_effect=run_sync
        ):
            agents_sdk.run()

        self.assertEqual(calls, 2)
        self.assertIsInstance(received_inputs[1], list)
        self.assertIn("Evidence gate", self.output.getvalue())
        self.assert_supported_output()


if __name__ == "__main__":
    unittest.main()
