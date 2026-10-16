import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini


class GeminiCachingTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs,
            CreateCachedContentConfig=lambda **kwargs: kwargs,
        )
        self.client.models.generate_content.return_value.usage_metadata = SimpleNamespace(
            prompt_token_count=2300,
            cached_content_token_count=2100,
            candidates_token_count=40,
        )

    def test_uncached_request_sends_policy_and_question(self):
        gemini.ask(self.client, self.types, "gemini-2.5-flash", "Can I change?", "policy")
        kwargs = self.client.models.generate_content.call_args.kwargs
        self.assertIn("policy", kwargs["contents"])
        self.assertIn("Can I change?", kwargs["contents"])
        self.assertIn("system_instruction", kwargs["config"])

    def test_cached_request_sends_only_question_and_cache_name(self):
        result = gemini.ask(self.client, self.types, "gemini-2.5-flash", "Can I change?", "policy", "cachedContents/123")
        kwargs = self.client.models.generate_content.call_args.kwargs
        self.assertNotIn("policy", kwargs["contents"])
        self.assertEqual(kwargs["config"]["cached_content"], "cachedContents/123")
        self.assertEqual(result.cached_tokens, 2100)

    def test_cache_creation_contains_policy_and_ttl(self):
        self.client.caches.create.return_value = SimpleNamespace(
            name="cachedContents/123", usage_metadata=SimpleNamespace(total_token_count=2200)
        )
        gemini.create_cache(self.client, self.types, "gemini-2.5-flash", "policy", "revision A")
        kwargs = self.client.caches.create.call_args.kwargs
        self.assertEqual(kwargs["config"]["contents"], ["policy"])
        self.assertEqual(kwargs["config"]["ttl"], "300s")


if __name__ == "__main__":
    unittest.main()
