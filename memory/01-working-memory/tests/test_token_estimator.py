"""Tests for tiktoken integration and its offline-safe fallback."""

from unittest.mock import patch

from token_estimator import TiktokenEstimator


def test_byte_fallback_counts_and_truncates_without_encoding_download() -> None:
    with patch("token_estimator.tiktoken.encoding_for_model", side_effect=OSError("offline")):
        estimator = TiktokenEstimator("test-model")
    assert estimator.estimate("12345678") == 8
    assert estimator.truncate("123456789", 4) == "1234"
