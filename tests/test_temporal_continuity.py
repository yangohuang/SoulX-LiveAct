import unittest

import torch

from temporal_continuity import anchor_chunk_start


class TemporalContinuityTests(unittest.TestCase):
    def test_anchor_blends_only_first_two_new_latents(self):
        clean = torch.tensor([[[[0.0]], [[3.0]], [[9.0]]]])
        previous = torch.tensor([[[[7.0]], [[10.0]]]])

        actual = anchor_chunk_start(clean, previous, 0.6)

        torch.testing.assert_close(actual, torch.tensor([[[[6.0]], [[4.4]], [[9.0]]]]))
        torch.testing.assert_close(clean, torch.tensor([[[[0.0]], [[3.0]], [[9.0]]]]))

    def test_zero_strength_preserves_baseline(self):
        clean = torch.randn(2, 3, 2, 2)
        previous = torch.randn(2, 2, 2, 2)

        self.assertIs(anchor_chunk_start(clean, previous, 0.0), clean)

    def test_invalid_strength_and_shape_are_rejected(self):
        clean = torch.randn(2, 3, 2, 2)
        previous = torch.randn(2, 2, 2, 2)
        for strength in (-0.1, 1.1):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                anchor_chunk_start(clean, previous, strength)
        with self.assertRaisesRegex(ValueError, "matching"):
            anchor_chunk_start(clean, torch.randn(3, 2, 2, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
