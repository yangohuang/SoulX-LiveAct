import hashlib
import unittest

import numpy as np
import torch

from frame_audit import exporter_frame_sha256, first_divergence
from util_liveact import get_audio_emb


class FrameAuditTests(unittest.TestCase):
    def test_hashes_exact_diffusers_uint8_export_input(self):
        frames = np.array([[[[0.0, 0.5, 1.0]]],
                           [[[1 / 255, 0.999, 0.25]]]], dtype=np.float32)
        expected = (frames * 255).astype(np.uint8)
        self.assertEqual(exporter_frame_sha256(frames), [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in expected])

    def test_first_divergence_and_equal_prefix(self):
        self.assertEqual(first_divergence(["a", "b", "c"], ["a", "b", "d"]), 2)
        self.assertIsNone(first_divergence(["a", "b"], ["a", "b", "c"]))
        self.assertEqual(first_divergence(["a"], ["b"]), 0)

    def test_request_end_audio_window_first_changes_at_block_22(self):
        canonical = torch.arange(720, dtype=torch.int32).view(720, 1, 1)
        repeated = canonical.repeat(3, 1, 1)
        mismatch = []
        for block in range(23):
            start = 0 if block <= 1 else (block - 1) * 32
            short = get_audio_emb(canonical, start, start + 53, "cpu")
            long = get_audio_emb(repeated, start, start + 53, "cpu")
            mismatch.append(int((short != long).sum()))

        self.assertEqual(mismatch[:22], [0] * 22)
        self.assertEqual(mismatch[22], 25)


if __name__ == "__main__":
    unittest.main()
