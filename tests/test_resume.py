import json
import tempfile
import unittest
from pathlib import Path

from dp_eval.cli import _preserve_invalid_attempt, _retryable_infrastructure_failure


class ResumeTests(unittest.TestCase):
    def test_provider_retry_failure_is_infrastructure_invalid(self):
        row = {"error": {"type": "RuntimeError", "message": "Provider request failed after retries: provider HTTP 429"}}
        self.assertTrue(_retryable_infrastructure_failure(row))
        self.assertTrue(_retryable_infrastructure_failure({"error": {"type": "ChunkedEncodingError", "message": "Response ended prematurely"}}))
        self.assertTrue(_retryable_infrastructure_failure({"error": {"type": "RuntimeError", "message": "RemoteDisconnected"}}))
        self.assertTrue(_retryable_infrastructure_failure({"error": {"type": "ItemTimeout", "message": "bounded timeout"}}))
        self.assertFalse(_retryable_infrastructure_failure({"error": {"type": "ValueError", "message": "bad law"}}))

    def test_invalid_attempt_is_preserved_without_nesting_old_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "seed-0"
            trial.mkdir()
            (trial / "trial.json").write_text("old")
            _preserve_invalid_attempt(trial)
            self.assertEqual((trial / "invalid-infrastructure-attempt" / "trial.json").read_text(), "old")
            (trial / "trial.json").write_text("new failure")
            _preserve_invalid_attempt(trial)
            self.assertEqual((trial / "invalid-infrastructure-attempt-2" / "trial.json").read_text(), "new failure")
            self.assertEqual((trial / "invalid-infrastructure-attempt" / "trial.json").read_text(), "old")


if __name__ == "__main__":
    unittest.main()
