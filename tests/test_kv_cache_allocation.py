import unittest

import torch

import runtime_options


class KvCacheAllocationTests(unittest.TestCase):
    def test_fp8_cpu_cache_without_audio_guidance(self):
        self.assertTrue(hasattr(runtime_options, "allocate_kv_caches"))
        caches, null_caches = runtime_options.allocate_kv_caches(
            token_count=2,
            step_count=3,
            layer_count=2,
            device="cpu",
            fp8=True,
            mean_memory=False,
            offload=True,
            audio_cfg=1.0,
        )

        self.assertIsNone(null_caches)
        self.assertEqual(set(caches), {0, 1, 2})
        self.assertEqual(set(caches[0]), {0, 1})
        entry = caches[0][0]
        self.assertEqual(entry["k"].shape, (1, 2, 40, 128))
        self.assertEqual(entry["v"].dtype, torch.float8_e4m3fn)
        self.assertEqual(entry["k_scale"].shape, (1, 2, 40, 1))
        self.assertEqual(entry["k_scale"].dtype, torch.float32)
        self.assertTrue(entry["offload_cache"])
        self.assertTrue(entry["fp8_kv_cache"])

    def test_audio_guidance_has_independent_bf16_caches(self):
        self.assertTrue(hasattr(runtime_options, "allocate_kv_caches"))
        caches, null_caches = runtime_options.allocate_kv_caches(
            token_count=2,
            step_count=2,
            layer_count=1,
            device="cpu",
            fp8=False,
            mean_memory=True,
            offload=True,
            audio_cfg=1.7,
        )

        self.assertIsNotNone(null_caches)
        self.assertEqual(caches[0][0]["k"].dtype, torch.bfloat16)
        self.assertIsNone(caches[0][0]["k_scale"])
        self.assertTrue(caches[0][0]["mean_memory"])
        caches[0][0]["k"][0, 0, 0, 0] = 1
        self.assertEqual(float(null_caches[0][0]["k"][0, 0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
