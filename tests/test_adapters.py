import unittest

from dp_eval.adapters import _content_text, _require_exact_model


class AdapterTests(unittest.TestCase):
    def test_content_text_variants(self):
        self.assertEqual(_content_text("x"), "x")
        self.assertEqual(_content_text([{"text": "a"}, {"text": "b"}]), "ab")
        self.assertEqual(_content_text(None), "")

    def test_exact_model_required(self):
        _require_exact_model("claude-opus-5", "claude-opus-5")
        with self.assertRaises(RuntimeError):
            _require_exact_model("claude-opus-5", "claude-opus-4-6")


if __name__ == "__main__":
    unittest.main()
