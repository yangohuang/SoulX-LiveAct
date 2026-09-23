import unittest

import torch

from model_liveact.model_memory import WanSelfAttention
from model_liveact.model_memory_sp import WanSelfAttention as WanSelfAttentionSP


class KvCacheTransferTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_to_cpu_cache_does_not_pin_every_layer(self):
        for attention_class in (WanSelfAttention, WanSelfAttentionSP):
            with self.subTest(attention_class=attention_class.__module__):
                attention = attention_class(dim=8, num_heads=2)
                cache = {
                    key: torch.ones(1024 * 1024, dtype=torch.uint8, device="cuda")
                    for key in ("k", "v", "k_scale", "v_scale")
                }

                attention._move_kv_cache_to_device(cache, "cpu")
                torch.cuda.synchronize()

                for value in cache.values():
                    self.assertEqual(value.device.type, "cpu")
                    self.assertFalse(value.is_pinned())


if __name__ == "__main__":
    unittest.main()
