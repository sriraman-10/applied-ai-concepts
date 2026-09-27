"""Deterministic payment-support tools for the episodic-memory demo."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "check_transaction",
        "description": "Check the current transaction state for an exact transaction ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_gateway",
        "description": "Inspect the payment gateway result for an exact transaction ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_refund",
        "description": "Check refund state before considering any refund action.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "issue_refund",
        "description": "Issue a refund only after checks prove that no refund exists or is pending.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "resolve_case",
        "description": "Mark the case resolved after required checks; provide the factual resolution.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"resolution": {"type": "string"}},
            "required": ["resolution"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "escalate_case",
        "description": "Escalate when required information is missing or the case cannot be safely resolved.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


class SupportToolError(ValueError):
    pass


class SupportTools:
    def __init__(self) -> None:
        self._transactions = {
            "TXN-DEMO-001": {
                "transaction": {"state": "failed", "amount_deducted": True},
                "gateway": {"result": "timeout", "captured": False},
                "refund": {"state": "auto_refund_triggered", "eta_business_days": 5},
            },
            "TXN-PAID-002": {
                "transaction": {"state": "succeeded", "amount_deducted": True},
                "gateway": {"result": "captured", "captured": True},
                "refund": {"state": "none"},
            },
            "TXN-REFUNDED-003": {
                "transaction": {"state": "failed", "amount_deducted": True},
                "gateway": {"result": "declined", "captured": False},
                "refund": {
                    "state": "completed",
                    "eta_business_days": None,
                    "customer_guidance": (
                        "The refund is completed. Escalate if the customer reports that it "
                        "is not visible."
                    ),
                },
            },
        }

    def _case(self, transaction_id: str) -> dict[str, Any]:
        try:
            return self._transactions[transaction_id]
        except KeyError as error:
            raise SupportToolError(
                "transaction not found; escalate for manual lookup"
            ) from error

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "check_transaction":
            return {
                "transaction_id": arguments["transaction_id"],
                **deepcopy(self._case(arguments["transaction_id"])["transaction"]),
            }
        if name == "check_gateway":
            return {
                "transaction_id": arguments["transaction_id"],
                **deepcopy(self._case(arguments["transaction_id"])["gateway"]),
            }
        if name == "check_refund":
            return {
                "transaction_id": arguments["transaction_id"],
                **deepcopy(self._case(arguments["transaction_id"])["refund"]),
            }
        if name == "issue_refund":
            transaction_id = arguments["transaction_id"]
            case = self._case(transaction_id)
            if case["refund"]["state"] in {
                "auto_refund_triggered",
                "pending",
                "completed",
            }:
                return {
                    "issued": False,
                    "reason": "refund already exists",
                    "state": case["refund"]["state"],
                }
            case["refund"] = {"state": "pending", "eta_business_days": 5}
            return {"issued": True, "state": "pending", "eta_business_days": 5}
        if name == "resolve_case":
            return {"resolved": True, "resolution": arguments["resolution"]}
        if name == "escalate_case":
            return {"escalated": True, "reason": arguments["reason"]}
        raise SupportToolError(f"unknown support tool: {name}")
