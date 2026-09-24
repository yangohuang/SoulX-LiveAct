import unittest
from unittest import mock

import torch

from model_liveact.model_memory import WanSelfAttention
from model_liveact.model_memory_sp import WanSelfAttention as WanSelfAttentionSP


class KvCacheTransferTests(unittest.TestCase):
    def test_default_gpu_to_cpu_copy_remains_non_blocking(self):
        for attention_class in (WanSelfAttention, WanSelfAttentionSP):
            with self.subTest(attention_class=attention_class.__module__):
                attention = attention_class(dim=8, num_heads=2)
                tensor = mock.Mock()
                tensor.to.return_value = tensor
                cache = {"k": tensor, "v": tensor, "k_scale": None, "v_scale": None}

                attention._move_kv_cache_to_device(cache, "cpu")

                self.assertEqual(tensor.to.call_args_list, [
                    mock.call(device="cpu", non_blocking=True),
                    mock.call(device="cpu", non_blocking=True),
                ])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_to_cpu_cache_does_not_pin_every_layer(self):
        for attention_class in (WanSelfAttention, WanSelfAttentionSP):
            with self.subTest(attention_class=attention_class.__module__):
                attention = attention_class(dim=8, num_heads=2)
                cache = {
                    key: torch.ones(1024 * 1024, dtype=torch.uint8, device="cuda")
                    for key in ("k", "v", "k_scale", "v_scale")
                }
                cache["blocking_cpu_copy"] = True

                attention._move_kv_cache_to_device(cache, "cpu")
                torch.cuda.synchronize()

                for key in ("k", "v", "k_scale", "v_scale"):
                    value = cache[key]
                    self.assertEqual(value.device.type, "cpu")
                    self.assertFalse(value.is_pinned())


if __name__ == "__main__":
    unittest.main()
