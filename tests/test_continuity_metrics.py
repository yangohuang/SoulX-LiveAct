import unittest

import numpy as np

from continuity_metrics import boundary_metrics


class BoundaryMetricsTests(unittest.TestCase):
    def test_separates_boundary_and_later_transitions(self):
        # Frame differences are 1 except for the first transition of block 1.
        values = np.arange(85, dtype=np.uint8)
        values[21:] += 9
        frames = np.broadcast_to(values[:, None, None], (85, 2, 2))

        result = boundary_metrics(frames, first_boundary=20, block_frames=32,
                                  opening_window=8)

        self.assertEqual(result["boundary_count"], 2)
        self.assertAlmostEqual(result["first_window_mean"], 1 + 9 / 16)
        self.assertAlmostEqual(result["later_window_mean"], 1)
        self.assertEqual(result["worst_seam"], 10)


if __name__ == "__main__":
    unittest.main()
