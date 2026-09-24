import unittest
from unittest import mock

import torch

import runtime_options


class KvCacheAllocationTests(unittest.TestCase):
    def test_blocking_cpu_copy_is_opt_in(self):
        for blocking in (False, True):
            with self.subTest(blocking=blocking):
                caches, _ = runtime_options.allocate_kv_caches(
                    token_count=2, step_count=1, layer_count=1, device="cpu",
                    fp8=True, mean_memory=False, offload=True, audio_cfg=1.0,
                    blocking_cpu_copy=blocking,
                )
                self.assertIs(caches[0][0]["blocking_cpu_copy"], blocking)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_first_step_can_reside_on_gpu(self):
        caches, null_caches = runtime_options.allocate_kv_caches(
            token_count=2, step_count=3, layer_count=1, device="cpu",
            fp8=True, mean_memory=False, offload=True, audio_cfg=1.0,
            resident_steps=1, onload_device="cuda:0",
        )
        self.assertIsNone(null_caches)
        self.assertEqual(caches[0][0]["k"].device.type, "cuda")
        self.assertEqual(caches[0][0]["k_scale"].device.type, "cuda")
        self.assertFalse(caches[0][0]["offload_cache"])
        for step in (1, 2):
            self.assertEqual(caches[step][0]["k"].device.type, "cpu")
            self.assertTrue(caches[step][0]["offload_cache"])

    def test_resident_step_requires_fp8_cpu_offload(self):
        for offload, fp8 in ((False, True), (True, False)):
            with self.subTest(offload=offload, fp8=fp8), self.assertRaises(ValueError):
                runtime_options.allocate_kv_caches(
                    token_count=2, step_count=3, layer_count=1, device="cpu",
                    fp8=fp8, mean_memory=False, offload=offload,
                    audio_cfg=1.0, resident_steps=1, onload_device="cuda:0")

    def test_resident_step_rejects_insufficient_vram(self):
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(0, 1)):
            with self.assertRaisesRegex(RuntimeError, "GPU memory"):
                runtime_options.allocate_kv_caches(
                    token_count=2, step_count=3, layer_count=1, device="cpu",
                    fp8=True, mean_memory=False, offload=True,
                    audio_cfg=1.0, resident_steps=1, onload_device="cuda:0")

    def test_resident_step_rejects_audio_cfg(self):
        with self.assertRaisesRegex(ValueError, "audio_cfg"):
            runtime_options.allocate_kv_caches(
                token_count=2, step_count=3, layer_count=1, device="cpu",
                fp8=True, mean_memory=False, offload=True,
                audio_cfg=1.7, resident_steps=1, onload_device="cuda:0")

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
