import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location("prompt_caching_demo", Path(__file__).resolve().parents[1] / "main.py")
module = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class PromptCachingTests(unittest.TestCase):
    def test_usage_and_prefix_order(self):
        usage = Mock(input_tokens=1500, output_tokens=40)
        usage.input_tokens_details.cached_tokens = 1280
        client = Mock()
        client.responses.create.return_value.usage = usage
        result = module.call(client, "gpt-4.1-mini", "fixed policy", "changing case")
        kwargs = client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["instructions"], "fixed policy")
        self.assertEqual(kwargs["input"], "changing case")
        self.assertFalse(kwargs["store"])
        self.assertEqual(kwargs["prompt_cache_key"], "northstar-travel-policy-demo")
        self.assertEqual(result.cached_tokens, 1280)

    def test_cached_tokens_are_subset_of_input(self):
        stats = module.summarize([
            module.Result(1.0, 1500, 0, 30),
            module.Result(0.5, 1510, 1280, 25),
        ])
        self.assertEqual(stats["input"], 3010)
        self.assertEqual(stats["uncached"], 1730)
        self.assertAlmostEqual(stats["hit_rate"], 1280 / 3010)


if __name__ == "__main__":
    unittest.main()
