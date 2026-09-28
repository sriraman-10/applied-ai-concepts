"""Deterministic support-system fixtures with hard refund idempotency."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class SupportToolError(RuntimeError):
    pass


class SupportTools:
    def __init__(self) -> None:
        self.transactions = {
            "TXN-DEMO-001": {
                "state": "failed",
                "amount": "49.00 USD",
                "order_created": False,
            },
            "TXN-DUP-002": {
                "state": "captured_twice",
                "amount": "29.00 USD",
                "order_created": True,
            },
            "TXN-REFUND-003": {
                "state": "refunded",
                "amount": "79.00 USD",
                "order_created": True,
            },
            "TXN-INCONSISTENT-004": {
                "state": "unknown",
                "amount": "19.00 USD",
                "order_created": False,
            },
        }
        self.gateways = {
            "TXN-DEMO-001": {"result": "timeout", "reference": "gw-demo-1"},
            "TXN-DUP-002": {"result": "duplicate_capture", "reference": "gw-demo-2"},
            "TXN-REFUND-003": {"result": "captured", "reference": "gw-demo-3"},
            "TXN-INCONSISTENT-004": {"result": "unavailable", "reference": None},
        }
        self.refunds = {
            "TXN-DEMO-001": {"state": "auto_refund_triggered", "eta_business_days": 5},
            "TXN-REFUND-003": {"state": "processing", "eta_business_days": 2},
        }
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _known(self, transaction_id: str) -> None:
        if transaction_id not in self.transactions:
            raise SupportToolError("transaction not found")

    def check_transaction(self, transaction_id: str) -> dict[str, Any]:
        self._known(transaction_id)
        return deepcopy(self.transactions[transaction_id])

    def check_gateway(self, transaction_id: str) -> dict[str, Any]:
        self._known(transaction_id)
        if transaction_id not in self.gateways:
            raise SupportToolError("gateway status unavailable")
        return deepcopy(self.gateways[transaction_id])

    def check_refund(self, transaction_id: str) -> dict[str, Any]:
        self._known(transaction_id)
        return deepcopy(
            self.refunds.get(
                transaction_id, {"state": "none", "eta_business_days": None}
            )
        )

    def issue_refund(self, transaction_id: str) -> dict[str, Any]:
        self._known(transaction_id)
        existing = self.refunds.get(transaction_id)
        if existing and existing["state"] not in {"failed", "cancelled"}:
            return {
                "issued": False,
                "reason": "refund already exists",
                **deepcopy(existing),
            }
        result = {"state": "submitted", "eta_business_days": 5}
        self.refunds[transaction_id] = result
        return {"issued": True, **deepcopy(result)}

    def escalate_case(self, reason: str) -> dict[str, Any]:
        return {"escalated": True, "reason": reason}

    def notify_customer(self, message: str) -> dict[str, Any]:
        if not message.strip():
            raise SupportToolError("customer message cannot be empty")
        return {"notified": True, "message": message}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "check_transaction",
            "check_gateway",
            "check_refund",
            "issue_refund",
            "escalate_case",
            "notify_customer",
        }
        if name not in allowed:
            raise SupportToolError(f"unknown support tool: {name}")
        result = getattr(self, name)(**arguments)
        self.events.append((name, deepcopy(arguments)))
        return result
