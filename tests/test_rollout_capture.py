import tempfile
import unittest
from pathlib import Path

import torch

from rollout_capture import save_rollout_block


class RolloutCaptureTests(unittest.TestCase):
    def test_saves_detached_cpu_copy_without_mutating_source(self):
        latent = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_rollout_block(Path(tmp), 3, latent)
            latent.fill_(999)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(path.name, "block-0003.pt")
        self.assertEqual(saved["block_index"], 3)
        self.assertEqual(saved["latent"].device.type, "cpu")
        torch.testing.assert_close(saved["latent"], torch.arange(48).reshape(2, 3, 2, 4).float())

    def test_rejects_invalid_index_or_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                save_rollout_block(Path(tmp), -1, torch.zeros(2, 3, 4, 5))
            with self.assertRaisesRegex(ValueError, "four dimensions"):
                save_rollout_block(Path(tmp), 0, torch.zeros(2, 3, 4))


if __name__ == "__main__":
    unittest.main()
