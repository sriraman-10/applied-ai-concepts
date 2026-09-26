"""Offline tests for deterministic incident evidence."""

import unittest

from tools import call_tool, get_dependency_health, get_logs, get_recent_deployments, get_service_metrics


class IncidentEvidenceTests(unittest.TestCase):
    def test_expected_investigation_path(self) -> None:
        checkout = get_service_metrics("checkout-service")
        self.assertEqual(checkout["p95_latency_ms"], 1850)
        self.assertEqual(checkout["database_latency_ms"], 18)
        self.assertEqual(get_recent_deployments("checkout-service")["deployments_last_24h"], [])

        checkout_dependencies = get_dependency_health("checkout-service")
        self.assertIn(
            {"service": "payment-service", "status": "degraded", "p95_latency_ms": 1720},
            checkout_dependencies["dependencies"],
        )
        payment_dependencies = get_dependency_health("payment-service")
        self.assertIn(
            {"service": "fraud-service", "status": "degraded", "p95_latency_ms": 1680},
            payment_dependencies["dependencies"],
        )
        payment_logs = get_logs("payment-service", "timeout fraud-service")
        self.assertEqual(payment_logs["entries"][0]["dependency"], "fraud-service")
        self.assertEqual(get_service_metrics("fraud-service")["p95_latency_ms"], 1680)

    def test_dispatch_errors_are_structured(self) -> None:
        self.assertIn("error", call_tool("unknown", {}))
        self.assertIn("error", call_tool("get_logs", {"service": "payment-service"}))


if __name__ == "__main__":
    unittest.main()
