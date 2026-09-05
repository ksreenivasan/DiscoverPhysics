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

    def test_missing_manifest_item_is_explicit_and_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            trial_dir = run_dir / "raw" / "lane" / "gravity" / "seed-0"
            trial_dir.mkdir(parents=True)
            (trial_dir / "trial.json").write_text(json.dumps({
                "item_key": "lane/gravity/seed-0", "lane": "lane",
                "model_id": "served-id", "provider": "openai_compatible",
                "world": "gravity", "seed": 0, "status": "completed",
                "joint_pass": True, "normalized_mse": 0.01,
                "explanation_score": 1.0,
            }))
            items = [
                {"item_key": f"lane/gravity/seed-{seed}", "lane": "lane",
                 "model_id": "served-id", "provider": "openai_compatible",
                 "world": "gravity", "seed": seed}
                for seed in range(2)
            ]
            (run_dir / "manifest.json").write_text(json.dumps({
                "trial_count": 2, "items": items,
            }))

            result = aggregate(run_dir)

            self.assertFalse(result["run_complete"])
            self.assertEqual(result["unresolved_missing_items"], ["lane/gravity/seed-1"])
            self.assertEqual(result["lanes"]["lane"]["trial_count"], 2)
            self.assertEqual(result["lanes"]["lane"]["failed_trials"], 1)

    def test_all_missing_manifest_items_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            run_dir.mkdir()
            item = {
                "item_key": "lane/gravity/seed-0", "lane": "lane",
                "model_id": "served-id", "provider": "openai_compatible",
                "world": "gravity", "seed": 0,
            }
            (run_dir / "manifest.json").write_text(json.dumps({
                "trial_count": 1, "items": [item],
            }))

            result = aggregate(run_dir)

            self.assertFalse(result["run_complete"])
            self.assertEqual(result["unresolved_missing_items"], [item["item_key"]])
            self.assertEqual(result["lanes"]["lane"]["trial_count"], 1)

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
