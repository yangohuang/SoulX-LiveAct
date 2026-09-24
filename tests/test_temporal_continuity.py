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

    def test_trend_extrapolates_previous_motion_only_at_chunk_start(self):
        clean = torch.tensor([[[[0.0]], [[3.0]], [[9.0]]]])
        previous = torch.tensor([[[[7.0]], [[10.0]]]])

        actual = anchor_chunk_start(clean, previous, 0.6, trend=0.5)

        torch.testing.assert_close(actual, torch.tensor([[[[6.9]], [[5.0]], [[9.0]]]]))
        torch.testing.assert_close(clean, torch.tensor([[[[0.0]], [[3.0]], [[9.0]]]]))

    def test_zero_trend_matches_fixed_anchor(self):
        clean = torch.randn(2, 3, 2, 2)
        previous = torch.randn(2, 2, 2, 2)

        torch.testing.assert_close(anchor_chunk_start(clean, previous, 0.25, trend=0),
                                   anchor_chunk_start(clean, previous, 0.25))

    def test_trend_requires_two_previous_latents(self):
        clean = torch.randn(2, 3, 2, 2)
        previous = torch.randn(2, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "two previous"):
            anchor_chunk_start(clean, previous, 0.25, trend=0.5)

    def test_invalid_strength_and_shape_are_rejected(self):
        clean = torch.randn(2, 3, 2, 2)
        previous = torch.randn(2, 2, 2, 2)
        for strength in (-0.1, 1.1):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                anchor_chunk_start(clean, previous, strength)
        for trend in (-0.1, 1.1):
            with self.subTest(trend=trend), self.assertRaisesRegex(ValueError, "trend"):
                anchor_chunk_start(clean, previous, 0.25, trend=trend)
        with self.assertRaisesRegex(ValueError, "matching"):
            anchor_chunk_start(clean, torch.randn(3, 2, 2, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
