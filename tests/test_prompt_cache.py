import tempfile
import unittest
from pathlib import Path

import torch

from prompt_cache import load_prompt_cache, save_prompt_cache


class PromptCacheTests(unittest.TestCase):
    def test_roundtrip_preserves_contexts_and_edit_prompts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "t5.bin"
            source.write_bytes(b"t5-v1")
            cached = {"prompt-a": ([torch.arange(4, dtype=torch.bfloat16)],
                                   {"(1, 2)": torch.ones(2, dtype=torch.bfloat16)})}
            save_prompt_cache(cached, root / "cache", [source])
            loaded = load_prompt_cache(root / "cache", [source], ["prompt-a"])
            self.assertTrue(torch.equal(loaded["prompt-a"][0][0], cached["prompt-a"][0][0]))
            self.assertTrue(torch.equal(loaded["prompt-a"][1]["(1, 2)"],
                                        cached["prompt-a"][1]["(1, 2)"]))

    def test_new_prompt_rejects_old_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "t5.bin"
            source.write_bytes(b"t5-v1")
            save_prompt_cache({"prompt-a": ([torch.ones(2)], {})}, root / "cache", [source])
            with self.assertRaisesRegex(ValueError, "prompt catalog changed"):
                load_prompt_cache(root / "cache", [source], ["prompt-b"])


if __name__ == "__main__":
    unittest.main()
