import unittest
from pathlib import Path

from dp_eval.cli import validate_full_prep


class FullPrepTests(unittest.TestCase):
    def test_template_is_valid_but_not_launchable(self):
        result = validate_full_prep(Path("/app/configs/full.template.yaml"))
        self.assertTrue(result["template_valid"])
        self.assertFalse(result["ready_to_launch"])
        self.assertTrue(result["checks"]["private_worlds_deliberately_empty"])
        self.assertEqual(result["checks"]["max_tokens"], 16384)


if __name__ == "__main__":
    unittest.main()
