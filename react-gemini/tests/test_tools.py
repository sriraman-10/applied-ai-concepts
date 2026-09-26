"""Offline checks for the deterministic incident evidence."""

import unittest

from tools import (
    get_dependency_health,
    get_logs,
    get_recent_deployments,
    get_service_metrics,
)


class IncidentEvidenceTests(unittest.TestCase):
    def test_expected_investigation_path(self) -> None:
        checkout = get_service_metrics("checkout-service")
        self.assertEqual(checkout["p95_latency_ms"], 1850)
        self.assertEqual(checkout["cpu_percent"], 42)
        self.assertEqual(checkout["database_latency_ms"], 18)

        deployments = get_recent_deployments("checkout-service")
        self.assertEqual(deployments["deployments_last_24h"], [])

        checkout_dependencies = get_dependency_health("checkout-service")
        self.assertIn(
            {"service": "payment-service", "status": "degraded", "p95_latency_ms": 1720},
            checkout_dependencies["dependencies"],
        )

        payment_logs = get_logs("payment-service", "timeout fraud-service")
        self.assertEqual(payment_logs["entries"][0]["dependency"], "fraud-service")

        fraud = get_service_metrics("fraud-service")
        self.assertEqual(fraud["p95_latency_ms"], 1680)


if __name__ == "__main__":
    unittest.main()
