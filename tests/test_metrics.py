import json
import tempfile
import unittest
from pathlib import Path

from dp_eval.metrics import aggregate


class MetricsTests(unittest.TestCase):
    def test_exact_pass_at_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            for world in ("gravity", "oscillator"):
                for seed in range(5):
                    trial_dir = run_dir / "raw" / "lane" / world / f"seed-{seed}"
                    trial_dir.mkdir(parents=True)
                    passed = seed == 0
                    (trial_dir / "trial.json").write_text(json.dumps({
                        "lane": "lane", "model_id": "gemini-3.7-flash", "provider": "google",
                        "provider_backend": "native", "world": world, "seed": seed,
                        "status": "completed", "joint_pass": passed,
                        "normalized_mse": 0.01 if passed else 1.0,
                        "explanation_score": 1.0 if passed else 0.0,
                    }))
            row = aggregate(run_dir)["lanes"]["lane"]
            self.assertAlmostEqual(row["pass_at_1"], 20.0)
            self.assertAlmostEqual(row["pass_at_3"], 60.0)
            self.assertAlmostEqual(row["pass_at_5"], 100.0)

    def test_failed_attempt_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            for seed in range(5):
                trial_dir = run_dir / "raw" / "lane" / "gravity" / f"seed-{seed}"
                trial_dir.mkdir(parents=True)
                (trial_dir / "trial.json").write_text(json.dumps({
                    "lane": "lane", "model_id": "gpt-5.6-sol", "provider": "openai",
                    "provider_backend": "native", "world": "gravity", "seed": seed,
                    "status": "failed", "joint_pass": False, "normalized_mse": None,
                    "explanation_score": None,
                }))
            row = aggregate(run_dir)["lanes"]["lane"]
            self.assertEqual(row["failed_trials"], 5)
            self.assertEqual(row["pass_at_5"], 0.0)


if __name__ == "__main__":
    unittest.main()
