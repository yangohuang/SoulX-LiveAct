import unittest

import numpy as np

from audio_conditioning_audit import compare_periods


class AudioConditioningAuditTests(unittest.TestCase):
    def test_identical_interiors_pass_despite_different_edges(self):
        one = np.tile([1.0, 0.0], (6, 1, 1))
        two = one.copy()
        two[0] = [0.0, 1.0]
        two[-1] = [0.0, 1.0]
        result = compare_periods(one, two, margin_frames=1)
        self.assertEqual(result["frames"], 4)
        self.assertAlmostEqual(result["mean_cosine"], 1.0)

    def test_different_interior_does_not_pass(self):
        one = np.tile([1.0, 0.0], (6, 1, 1))
        two = np.tile([0.0, 1.0], (6, 1, 1))
        result = compare_periods(one, two, margin_frames=1)
        self.assertAlmostEqual(result["mean_cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()
