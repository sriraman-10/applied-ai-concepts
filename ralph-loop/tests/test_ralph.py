"""Offline integration checks with scripted model decisions, never API calls."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ralph

# Test fixture only; the orchestrator never receives a built-in LIS algorithm.
GOOD_CODE = '''from itertools import combinations
def length_of_lis(nums: list[int]) -> int:
    return max(len(s) for n in range(len(nums) + 1)
               for s in combinations(nums, n)
               if all(a < b for a, b in zip(s, s[1:])))
'''


def decision(code="", status="IMPROVE"):
    return dict(status=status, approach_name="Test approach", explanation="Test decision",
                time_complexity="Test time", space_complexity="Test space", code=code)


class RalphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ralph.ROOT)
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        (self.workspace / "solutions").mkdir()
        for name in ("problem.md", "progress.md"):
            (self.workspace / name).write_text(name, encoding="utf-8")
        patcher = patch.object(ralph, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_decisions(self, decisions, limit=10):
        requests = []
        responses = iter(decisions)

        def generate_content(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text=json.dumps(next(responses)))

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.object(ralph, "MAX_ITERATIONS", limit), redirect_stdout(io.StringIO()):
            complete = ralph.run(client)
        return complete, requests, (self.workspace / "progress.md").read_text()

    def test_failure_then_repair_then_completion(self):
        complete, requests, progress = self.run_decisions([
            decision("def length_of_lis(nums): return 999"),
            decision(GOOD_CODE), decision(status="COMPLETE"),
        ])
        self.assertTrue(complete)
        self.assertIn("FAIL", progress)
        self.assertIn("STATUS: COMPLETE", progress)
        self.assertNotIn("solution_v1.py", requests[0]["contents"])
        self.assertIn("solution_v1.py", requests[1]["contents"])
        self.assertIn("FAIL", requests[1]["contents"])
        self.assertIn("solution_v2.py", requests[2]["contents"])
        self.assertTrue(all(isinstance(r["contents"], str) for r in requests))

    def test_false_completion_and_bound(self):
        complete, _, progress = self.run_decisions([decision(status="COMPLETE")], limit=1)
        self.assertFalse(complete)
        self.assertIn("Completion rejected", progress)
        self.assertIn("STATUS: MAX_ITERATIONS_REACHED", progress)

    def test_resumption_preserves_files(self):
        self.run_decisions([decision(GOOD_CODE)], limit=1)
        old = (self.workspace / "solutions/solution_v1.py").read_bytes()
        self.run_decisions([decision(GOOD_CODE)], limit=1)
        self.assertEqual(old, (self.workspace / "solutions/solution_v1.py").read_bytes())
        self.assertTrue((self.workspace / "solutions/solution_v2.py").exists())
        complete, _, _ = self.run_decisions([decision(status="COMPLETE")])
        self.assertTrue(complete)

    def test_api_error_diagnostics_redact_keys(self):
        error = Exception("request rejected")
        error.code = 429
        error.status = "RESOURCE_EXHAUSTED"
        error.message = "Quota exceeded for secret-test-key and AIzaExampleKey123"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-test-key"}):
            details = ralph.describe_error(error)
        self.assertIn("429 RESOURCE_EXHAUSTED", details)
        self.assertIn("Quota exceeded", details)
        self.assertIn("billing", details)
        self.assertNotIn("secret-test-key", details)
        self.assertNotIn("AIzaExampleKey123", details)

    def test_invalid_response_is_persisted(self):
        complete, _, progress = self.run_decisions([{"status": "UNKNOWN"}], limit=1)
        self.assertFalse(complete)
        self.assertIn("Invalid response", progress)

    def test_verifier_reports_errors_and_timeout(self):
        path = self.workspace / "solutions/broken.py"
        for code in ("this is invalid python!", "raise RuntimeError('broken')",
                     "def length_of_lis(nums): return True"):
            path.write_text(code)
            passed, details = ralph.verify(path)
            self.assertFalse(passed)
            self.assertIn("FAIL", details)
        path.write_text("while True: pass")
        with patch.object(ralph, "VERIFY_TIMEOUT", 0.1):
            passed, details = ralph.verify(path)
        self.assertFalse(passed)
        self.assertIn("exceeded", details)


if __name__ == "__main__":
    unittest.main()
