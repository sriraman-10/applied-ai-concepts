"""Idempotent, application-owned seed procedure definitions."""

from __future__ import annotations

from memory import ProceduralMemoryManager
from models import ProcedureStatus


SEEDS = (
    dict(
        procedure_key="FAILED_PAYMENT_PLAYBOOK",
        name="Failed Payment Resolution",
        description="Approved procedure for failed or timed-out payment transactions and charged checkouts without an order.",
        category="payments",
        issue_patterns=(
            "payment failed",
            "checkout failed",
            "money deducted but order failed",
            "card charged without order",
        ),
        steps=(
            {"action": "identify_transaction"},
            {"action": "check_transaction", "tool": "check_transaction"},
            {"action": "check_gateway", "tool": "check_gateway"},
            {"action": "check_refund", "tool": "check_refund"},
            {"action": "prevent_duplicate_refund"},
            {"action": "notify_customer", "tool": "notify_customer"},
        ),
        preconditions=("customer supplies a transaction identifier",),
        completion_conditions=(
            "transaction status known",
            "gateway result known",
            "refund status known",
            "customer informed",
        ),
        escalation_rules=(
            "transaction state inconsistent",
            "gateway status unavailable",
            "transaction not found",
        ),
    ),
    dict(
        procedure_key="DUPLICATE_CHARGE_PLAYBOOK",
        name="Duplicate Charge Resolution",
        description="Approved procedure for duplicate card captures or two charges for one order.",
        category="payments",
        issue_patterns=(
            "charged twice",
            "duplicate card charge",
            "two payments for one order",
        ),
        steps=(
            {"action": "identify_transaction"},
            {"action": "check_transaction", "tool": "check_transaction"},
            {"action": "check_gateway", "tool": "check_gateway"},
            {"action": "check_refund", "tool": "check_refund"},
            {"action": "issue_refund", "tool": "issue_refund"},
            {"action": "notify_customer", "tool": "notify_customer"},
        ),
        preconditions=("duplicate capture is reported with a transaction identifier",),
        completion_conditions=(
            "duplicate capture verified",
            "refund status known",
            "customer informed",
        ),
        escalation_rules=(
            "captures do not share an order",
            "gateway status unavailable",
            "refund submission fails",
        ),
    ),
    dict(
        procedure_key="REFUND_DELAY_PLAYBOOK",
        name="Refund Delay Investigation",
        description="Approved procedure for a refund that exists but has not reached the customer.",
        category="refunds",
        issue_patterns=(
            "refund delayed",
            "refund pending",
            "refund not received",
            "waiting for refund",
        ),
        steps=(
            {"action": "identify_transaction"},
            {"action": "check_transaction", "tool": "check_transaction"},
            {"action": "check_refund", "tool": "check_refund"},
            {"action": "prevent_duplicate_refund"},
            {"action": "notify_customer", "tool": "notify_customer"},
        ),
        preconditions=("a refund is expected for the transaction",),
        completion_conditions=(
            "refund status known",
            "verified ETA communicated when available",
            "customer informed",
        ),
        escalation_rules=(
            "refund record missing after confirmed issue",
            "refund exceeds verified ETA",
            "transaction not found",
        ),
    ),
)


async def seed_procedures(manager: ProceduralMemoryManager) -> list:
    created = []
    for item in SEEDS:
        existing = await manager.get_procedure(item["procedure_key"], 1)
        if existing is None:
            created.append(
                await manager.create_procedure(
                    **item,
                    status=ProcedureStatus.ACTIVE,
                    metadata={"approved_by": "demo-admin", "source": "seed"},
                )
            )
    return created
