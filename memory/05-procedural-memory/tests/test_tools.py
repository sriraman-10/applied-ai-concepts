import pytest
from tools import SupportToolError, SupportTools


def test_duplicate_refund_hard_safety():
    result = SupportTools().issue_refund("TXN-DEMO-001")
    assert result["issued"] is False and result["reason"] == "refund already exists"


def test_new_refund_is_idempotent():
    tools = SupportTools()
    first = tools.issue_refund("TXN-DUP-002")
    second = tools.issue_refund("TXN-DUP-002")
    assert first["issued"] is True and second["issued"] is False


def test_unknown_transaction_rejected():
    with pytest.raises(SupportToolError):
        SupportTools().check_transaction("UNKNOWN")


def test_arbitrary_tool_rejected():
    with pytest.raises(SupportToolError):
        SupportTools().execute("run_sql", {})
