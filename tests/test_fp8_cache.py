import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from fp8_cache import load_fp8_cache, save_fp8_cache
from fp8_gemm import FP8Linear, enable_fp8_gemm


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))])


class FP8CacheTests(unittest.TestCase):
    def test_roundtrip_skips_bf16_linear_weights(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.bin"
            source.write_bytes(b"source-v1")
            model = TinyModel().to(torch.bfloat16)
            enable_fp8_gemm(model)
            expected = {}
            for name, module in model.named_modules():
                if isinstance(module, FP8Linear):
                    module._fp8_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn).t()
                    module._fp8_weight_scale = torch.tensor([0.125], dtype=torch.float32)
                    module._fp16_weight_cpu = None
                    expected[name] = module._fp8_weight.clone()

            cache = root / "cache"
            save_fp8_cache(model, cache, [source])
            loaded = load_fp8_cache(TinyModel, cache, [source])

            for name, module in loaded.named_modules():
                if isinstance(module, FP8Linear):
                    self.assertTrue(torch.equal(module._fp8_weight, expected[name]))
                    self.assertEqual(module._fp8_weight.stride(0), 1)
                    self.assertIsNone(module._fp16_weight_cpu)
                    self.assertFalse(module.bias.is_meta)
            self.assertTrue(torch.equal(loaded.blocks[0][1].weight, model.blocks[0][1].weight))

    def test_changed_source_rejects_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.bin"
            source.write_bytes(b"source-v1")
            model = TinyModel()
            enable_fp8_gemm(model)
            for module in model.modules():
                if isinstance(module, FP8Linear):
                    module._fp8_weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
                    module._fp8_weight_scale = torch.ones(1)
                    module._fp16_weight_cpu = None
            cache = root / "cache"
            save_fp8_cache(model, cache, [source])
            source.write_bytes(b"source-v2")
            with self.assertRaisesRegex(ValueError, "source checkpoint changed"):
                load_fp8_cache(TinyModel, cache, [source])


if __name__ == "__main__":
    unittest.main()
