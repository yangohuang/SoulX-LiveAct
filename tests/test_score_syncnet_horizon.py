import unittest

import numpy as np

from score_syncnet_horizon import score_distances, score_segments


class ScoreSyncNetHorizonTests(unittest.TestCase):
    def test_each_segment_recomputes_shift_and_confidence(self):
        first = np.tile([5.0, 1.0, 4.0], (50, 1))
        second = np.tile([3.0, 2.0, 8.0], (48, 1))
        rows = score_segments(np.concatenate([first, second]), fps=25,
                              segment_seconds=2, segment_count=2, vshift=1)
        self.assertEqual([row["windows"] for row in rows], [50, 48])
        self.assertEqual([row["best_offset_25fps_frames"] for row in rows], [0, 0])
        self.assertAlmostEqual(rows[0]["confidence"], 3.0)
        self.assertAlmostEqual(rows[1]["confidence"], 1.0)

    def test_empty_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "segment"):
            score_segments(np.ones((25, 3)), fps=25,
                           segment_seconds=1, segment_count=2, vshift=1)

    def test_short_final_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "segment"):
            score_segments(np.ones((55, 3)), fps=25,
                           segment_seconds=2, segment_count=2, vshift=1)


if __name__ == "__main__":
    unittest.main()
