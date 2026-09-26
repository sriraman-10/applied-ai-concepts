"""Offline tests for evidence validation and terminal output."""

import io
import unittest
from unittest.mock import patch

from rich.console import Console

import terminal_ui
from terminal_ui import ToolEvent, evidence_verdict, missing_evidence
from tools import call_tool


def event(name: str, **arguments: str) -> ToolEvent:
    return ToolEvent(name, arguments, call_tool(name, arguments))


def complete_events() -> list[ToolEvent]:
    return [
        event("get_service_metrics", service="checkout-service"),
        event("get_recent_deployments", service="checkout-service"),
        event("get_dependency_health", service="checkout-service"),
        event("get_dependency_health", service="payment-service"),
        event("get_logs", service="payment-service", query="timeout fraud-service"),
        event("get_service_metrics", service="fraud-service"),
    ]


class VerdictTests(unittest.TestCase):
    def test_complete_evidence_supports_narrow_conclusion(self) -> None:
        events = complete_events()
        label, verdict = evidence_verdict(events)
        self.assertEqual(missing_evidence(events), [])
        self.assertEqual(label, "SUPPORTED")
        self.assertIn("fraud-service", verdict)
        self.assertIn("does not establish why", verdict)

    def test_call_alone_does_not_satisfy_a_requirement(self) -> None:
        bad_event = ToolEvent(
            "get_logs",
            {"service": "payment-service", "query": "fraud"},
            {"matches": 0, "entries": []},
        )
        missing = missing_evidence(complete_events()[:-2] + [bad_event, complete_events()[-1]])
        self.assertIn("payment-service timeout logs naming fraud-service", missing)

    def test_incomplete_evidence_is_inconclusive(self) -> None:
        label, verdict = evidence_verdict([event("get_service_metrics", service="checkout-service")])
        self.assertEqual(label, "INCONCLUSIVE")
        self.assertIn("complete evidence path", verdict)

    def test_final_output_separates_draft_from_verdict(self) -> None:
        output = io.StringIO()
        with patch.object(terminal_ui, "console", Console(file=output, force_terminal=False, width=120)):
            terminal_ui.show_final("CPU saturation is definitely the root cause.", complete_events())
        rendered = output.getvalue()
        self.assertIn("Model draft", rendered)
        self.assertIn("Verdict", rendered)
        self.assertIn("SUPPORTED", rendered)
        self.assertIn("does not establish why fraud-service itself is slow", rendered)

    def test_empty_run_has_explicit_evidence_row(self) -> None:
        output = io.StringIO()
        with patch.object(terminal_ui, "console", Console(file=output, force_terminal=False, width=100)):
            terminal_ui.show_final("", [])
        rendered = output.getvalue()
        self.assertIn("No tool evidence was collected", rendered)
        self.assertIn("INCONCLUSIVE", rendered)


if __name__ == "__main__":
    unittest.main()
