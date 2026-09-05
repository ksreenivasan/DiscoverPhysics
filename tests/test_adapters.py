import os
import unittest
from unittest.mock import Mock, patch

from dp_eval.adapters import (
    Adapter,
    HIGH_REASONING_MODELS,
    _apply_reasoning_history,
    _content_text,
    _require_exact_model,
)


class AdapterTests(unittest.TestCase):
    def test_content_text_variants(self):
        self.assertEqual(_content_text("x"), "x")
        self.assertEqual(_content_text([{"text": "a"}, {"text": "b"}]), "ab")
        self.assertEqual(_content_text(None), "")

    def test_exact_model_required(self):
        _require_exact_model("claude-opus-5", "claude-opus-5")
        with self.assertRaises(RuntimeError):
            _require_exact_model("claude-opus-5", "claude-opus-4-6")

    def test_gemini_38_requests_high_thinking(self):
        self.assertIn("gemini-3.8-flash", HIGH_REASONING_MODELS)

    def test_reasoning_history_policies(self):
        messages = [
            {"role": "assistant", "content": "visible", "reasoning_content": "hidden"},
            {"role": "user", "content": "next"},
        ]
        self.assertEqual(
            _apply_reasoning_history(messages, "preserve")[0]["reasoning_content"],
            "hidden",
        )
        self.assertNotIn(
            "reasoning_content", _apply_reasoning_history(messages, "none")[0]
        )
        self.assertEqual(
            _apply_reasoning_history(messages, "empty")[0]["reasoning_content"], ""
        )

    @patch.dict(os.environ, {"TEST_SERVED_KEY": "dummy-token"}, clear=False)
    @patch("dp_eval.adapters._request_with_retries")
    @patch("requests.get")
    def test_endpoint_canary_checks_catalog_then_inference(self, get, post):
        catalog = Mock(status_code=200)
        catalog.json.return_value = {"data": [{"id": "served-id"}]}
        get.return_value = catalog
        inference = Mock(headers={})
        inference.json.return_value = {
            "id": "response-id",
            "model": "served-id",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {},
        }
        post.return_value = inference
        adapter = Adapter(
            endpoint_config={
                "provider": "openai_compatible",
                "model_id": "served-id",
                "base_url": "http://model.example/v1/",
                "api_key_env": "TEST_SERVED_KEY",
                "reasoning_history": "empty",
            }
        )

        result = adapter.endpoint_canary()

        self.assertEqual(result["catalog"], "passed")
        self.assertEqual(result["inference"], "passed")
        self.assertEqual(get.call_args.args[0], "http://model.example/v1/models")
        self.assertEqual(
            post.call_args.args[0], "http://model.example/v1/chat/completions"
        )
        self.assertEqual(post.call_args.kwargs["payload"]["max_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
