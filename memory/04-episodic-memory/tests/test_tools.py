import pytest

from tools import SupportToolError, SupportTools


def test_deterministic_payment_fixture():
    tools = SupportTools()
    assert (
        tools.execute("check_transaction", {"transaction_id": "TXN-DEMO-001"})["state"]
        == "failed"
    )
    assert (
        tools.execute("check_gateway", {"transaction_id": "TXN-DEMO-001"})["result"]
        == "timeout"
    )
    assert (
        tools.execute("check_refund", {"transaction_id": "TXN-DEMO-001"})["state"]
        == "auto_refund_triggered"
    )


def test_duplicate_refund_is_prevented():
    result = SupportTools().execute("issue_refund", {"transaction_id": "TXN-DEMO-001"})
    assert result == {
        "issued": False,
        "reason": "refund already exists",
        "state": "auto_refund_triggered",
    }


def test_new_refund_becomes_idempotent():
    tools = SupportTools()
    first = tools.execute("issue_refund", {"transaction_id": "TXN-PAID-002"})
    second = tools.execute("issue_refund", {"transaction_id": "TXN-PAID-002"})
    assert first["issued"] is True
    assert second["issued"] is False


def test_unknown_transaction_is_safe_error():
    with pytest.raises(SupportToolError, match="not found"):
        SupportTools().execute("check_transaction", {"transaction_id": "UNKNOWN"})


def test_completed_refund_has_grounded_guidance_without_invented_eta():
    result = SupportTools().execute(
        "check_refund", {"transaction_id": "TXN-REFUNDED-003"}
    )
    assert result["state"] == "completed"
    assert result["eta_business_days"] is None
    assert "Escalate" in result["customer_guidance"]
