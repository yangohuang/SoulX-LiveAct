import unittest

import torch

from audio_embedding_repeat import period_fingerprints, repeat_embedding


class AudioEmbeddingRepeatTests(unittest.TestCase):
    def test_three_periods_and_interior_context_are_exact(self):
        source = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3)

        repeated = repeat_embedding(source, target_frames=18)
        fingerprints = period_fingerprints(repeated, period_frames=6,
                                           margin_frames=2, context_radius=1)

        self.assertEqual(tuple(repeated.shape), (18, 2, 3))
        self.assertTrue(torch.equal(repeated[:6], repeated[6:12]))
        self.assertTrue(torch.equal(repeated[:6], repeated[12:18]))
        self.assertEqual(len(set(fingerprints["period_sha256"])), 1)
        self.assertEqual(len(set(fingerprints["interior_context_sha256"])), 1)

    def test_noninteger_repeat_is_rejected(self):
        source = torch.ones((6, 2, 3), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "multiple"):
            repeat_embedding(source, target_frames=17)

    def test_margin_must_preserve_context(self):
        repeated = repeat_embedding(torch.ones((6, 2)), target_frames=18)
        with self.assertRaisesRegex(ValueError, "margin"):
            period_fingerprints(repeated, period_frames=6,
                                margin_frames=1, context_radius=2)


if __name__ == "__main__":
    unittest.main()
