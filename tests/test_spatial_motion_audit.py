import unittest

import numpy as np

from spatial_motion_audit import boundary_sums, measure_pair


class SpatialMotionAuditTests(unittest.TestCase):
    def test_textured_square_moves_only_in_lower_band(self):
        previous = np.zeros((180, 104), dtype=np.uint8)
        current = np.zeros_like(previous)
        checker = ((np.indices((24, 24)).sum(axis=0) % 2) * 180 + 40).astype(np.uint8)
        previous[130:154, 30:54] = checker
        current[130:154, 34:58] = checker

        measured = measure_pair(previous, current)

        self.assertGreater(measured["lower_body"]["mae"], 0)
        self.assertGreater(measured["lower_body"]["flow"], 0)
        self.assertEqual(measured["mouth"]["mae"], 0)
        self.assertEqual(measured["upper_head"]["mae"], 0)
        self.assertEqual(measured["background"]["mae"], 0)
        self.assertLess(measured["background"]["flow"], 1e-4)

    def test_exact_twenty_two_complete_chunk_windows(self):
        windows = boundary_sums(list(range(721)), frame_count=722)

        self.assertEqual(len(windows), 22)
        self.assertEqual(windows[0], {"index": 20, "sum": sum(range(20, 28))})
        self.assertEqual(windows[-1], {"index": 692, "sum": sum(range(692, 700))})


if __name__ == "__main__":
    unittest.main()
