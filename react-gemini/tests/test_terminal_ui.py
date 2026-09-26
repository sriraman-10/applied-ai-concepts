"""Offline checks for the shared evidence-based verdict."""

import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rich.console import Console

import gemini_sdk_react
import manual_react
import terminal_ui
from terminal_ui import ToolEvent, evidence_verdict
from tools import execute_tool


def event(name: str, **arguments: str) -> ToolEvent:
    return ToolEvent(name, arguments, execute_tool(name, arguments))


class VerdictTests(unittest.TestCase):
    def test_both_agents_share_the_minimum_evidence_contract(self) -> None:
        required_phrases = (
            "checkout-service metrics",
            "checkout-service dependency health",
            "payment-service dependency health",
            "payment-service logs",
            "fraud-service metrics",
        )
        for instructions in (manual_react.INSTRUCTIONS, gemini_sdk_react.INSTRUCTIONS):
            for phrase in required_phrases:
                self.assertIn(phrase, instructions)

    def test_complete_evidence_identifies_fraud_service(self) -> None:
        events = [
            event("get_service_metrics", service="checkout-service"),
            event("get_dependency_health", service="checkout-service"),
            event("get_dependency_health", service="payment-service"),
            event("get_logs", service="payment-service", query="timeout fraud-service"),
            event("get_service_metrics", service="fraud-service"),
        ]

        label, verdict, evidence = evidence_verdict(events)

        self.assertEqual(label, "SUPPORTED CONCLUSION")
        self.assertIn("fraud-service", verdict)
        self.assertIn("does not establish why", verdict)
        self.assertEqual(len(evidence), 5)

    def test_missing_evidence_is_inconclusive(self) -> None:
        events = [event("get_service_metrics", service="checkout-service")]

        label, verdict, evidence = evidence_verdict(events)

        self.assertEqual(label, "INCONCLUSIVE")
        self.assertIn("complete evidence path", verdict)
        self.assertEqual(len(evidence), 1)

    def test_unsupported_deeper_cause_is_rejected(self) -> None:
        events = [
            event("get_service_metrics", service="checkout-service"),
            event("get_dependency_health", service="checkout-service"),
            event("get_dependency_health", service="payment-service"),
            event("get_logs", service="payment-service", query="timeout fraud-service"),
            event("get_service_metrics", service="fraud-service"),
        ]
        output = io.StringIO()

        with patch.object(
            terminal_ui,
            "console",
            Console(file=output, force_terminal=False, width=120),
        ):
            terminal_ui.show_final("CPU saturation on fraud-service is the root cause.", events)

        rendered = output.getvalue()
        self.assertIn("UNSUPPORTED CLAIM", rendered)
        self.assertIn("does not establish why fraud-service itself is slow", rendered)
        self.assertNotIn("is the root cause", rendered)


class EntrypointOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.console_patch = patch.object(
            terminal_ui,
            "console",
            Console(file=self.output, force_terminal=False, width=120),
        )
        self.console_patch.start()
        self.addCleanup(self.console_patch.stop)

    def assert_correct_verdict(self) -> None:
        output = self.output.getvalue()
        self.assertIn("Phase 1 · Agent investigation", output)
        self.assertIn("Phase 2 · Evidence validation", output)
        self.assertIn("Evidence collected by the agent", output)
        self.assertIn("PASS", output)
        self.assertIn("fraud-service is the likely bottleneck", output)
        self.assertIn("underlying cause", output)
        self.assertNotIn("CPU saturation", output)

    def test_manual_loop_output_has_correct_verdict(self) -> None:
        calls = [
            SimpleNamespace(name="get_service_metrics", args={"service": "checkout-service"}, id="1"),
            SimpleNamespace(name="get_dependency_health", args={"service": "checkout-service"}, id="2"),
            SimpleNamespace(name="get_dependency_health", args={"service": "payment-service"}, id="3"),
            SimpleNamespace(name="get_logs", args={"service": "payment-service", "query": "timeout fraud-service"}, id="4"),
            SimpleNamespace(name="get_service_metrics", args={"service": "fraud-service"}, id="5"),
        ]
        responses = iter([
            SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace())],
                function_calls=calls,
                text=None,
            ),
            SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace())],
                function_calls=[],
                text=(
                    "fraud-service is the likely bottleneck. Its elevated latency propagates "
                    "through payment-service to checkout-service. The underlying cause of "
                    "fraud-service degradation is not established by the available evidence."
                ),
            ),
        ])
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **_: next(responses)),
            __enter__=lambda self: self,
            __exit__=lambda *_: None,
        )

        class FakeClient:
            def __enter__(self):
                return client

            def __exit__(self, *_):
                return None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch.object(
            manual_react.genai, "Client", return_value=FakeClient()
        ):
            manual_react.run()

        self.assert_correct_verdict()

    def test_sdk_loop_output_has_correct_verdict(self) -> None:
        class FakeChat:
            def send_message(self, message: str) -> SimpleNamespace:
                self.message = message
                gemini_sdk_react.get_service_metrics("checkout-service")
                gemini_sdk_react.get_dependency_health("checkout-service")
                gemini_sdk_react.get_dependency_health("payment-service")
                gemini_sdk_react.get_logs("payment-service", "timeout fraud-service")
                gemini_sdk_react.get_service_metrics("fraud-service")
                return SimpleNamespace(
                    text=(
                        "fraud-service is the likely bottleneck. Its elevated latency propagates "
                        "through payment-service to checkout-service. The underlying cause of "
                        "fraud-service degradation is not established by the available evidence."
                    )
                )

        chat = FakeChat()
        client = SimpleNamespace(chats=SimpleNamespace(create=lambda **_: chat))

        class FakeClient:
            def __enter__(self):
                return client

            def __exit__(self, *_):
                return None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch.object(
            gemini_sdk_react.genai, "Client", return_value=FakeClient()
        ):
            gemini_sdk_react.run()

        self.assert_correct_verdict()
        self.assertEqual(chat.message, gemini_sdk_react.USER_QUERY)

    def test_sdk_does_not_fill_in_missing_agent_evidence(self) -> None:
        messages = []

        def send_message(message: str) -> SimpleNamespace:
            messages.append(message)
            return SimpleNamespace(text="Insufficient tool evidence.")

        chat = SimpleNamespace(send_message=send_message)
        client = SimpleNamespace(chats=SimpleNamespace(create=lambda **_: chat))

        class FakeClient:
            def __enter__(self):
                return client

            def __exit__(self, *_):
                return None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch.object(
            gemini_sdk_react.genai, "Client", return_value=FakeClient()
        ):
            gemini_sdk_react.run()

        output = self.output.getvalue()
        self.assertIn("INCOMPLETE", output)
        self.assertIn("payment-service logs", output)
        self.assertEqual(gemini_sdk_react.EVENTS, [])
        self.assertEqual(len(messages), gemini_sdk_react.MAX_INVESTIGATION_ROUNDS)

    def test_sdk_requests_continuation_then_completes(self) -> None:
        messages = []

        def send_message(message: str) -> SimpleNamespace:
            messages.append(message)
            if len(messages) == 1:
                return SimpleNamespace(text="The likely cause is downstream.")
            gemini_sdk_react.get_service_metrics("checkout-service")
            gemini_sdk_react.get_dependency_health("checkout-service")
            gemini_sdk_react.get_dependency_health("payment-service")
            gemini_sdk_react.get_logs("payment-service", "timeout fraud-service")
            gemini_sdk_react.get_service_metrics("fraud-service")
            return SimpleNamespace(
                text=(
                    "fraud-service is the likely bottleneck. The underlying cause "
                    "of its degradation is not established."
                )
            )

        chat = SimpleNamespace(send_message=send_message)
        client = SimpleNamespace(chats=SimpleNamespace(create=lambda **_: chat))

        class FakeClient:
            def __enter__(self):
                return client

            def __exit__(self, *_):
                return None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch.object(
            gemini_sdk_react.genai, "Client", return_value=FakeClient()
        ):
            gemini_sdk_react.run()

        output = self.output.getvalue()
        self.assertEqual(len(messages), 2)
        self.assertIn("Agent continuation", output)
        self.assertIn("PASS", output)

    def test_sdk_version_mismatch_has_clear_error(self) -> None:
        with patch.object(gemini_sdk_react, "version", return_value="1.75.0"):
            with self.assertRaisesRegex(SystemExit, "requires google-genai 2.x"):
                gemini_sdk_react._require_supported_sdk()

    def test_unevaluated_function_call_has_clear_error(self) -> None:
        part = SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(name="get_service_metrics", args={"service": "checkout-service"}),
        )
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        )

        with self.assertRaisesRegex(RuntimeError, "get_service_metrics"):
            gemini_sdk_react._final_text(response)


if __name__ == "__main__":
    unittest.main()
