import unittest

import numpy as np

from mouth_motion_proxy import region_changes, summarize_periods


class MouthMotionProxyTests(unittest.TestCase):
    def test_mouth_change_is_separate_from_upper_face_change(self):
        frames = np.zeros((4, 224, 224), dtype=np.uint8)
        frames[1, 90:150, 60:160] = 20
        frames[2, 90:150, 60:160] = 40
        mouth, upper = region_changes(frames)
        self.assertEqual(mouth.tolist(), [20.0, 20.0, 40.0])
        self.assertEqual(upper.tolist(), [0.0, 0.0, 0.0])

    def test_period_summary_excludes_transition_between_repetitions(self):
        changes = np.array([1.0, 1.0, 100.0, 2.0, 2.0])
        rows = summarize_periods(changes, np.zeros_like(changes), fps=1,
                                 segment_seconds=3, segment_count=2)
        self.assertEqual([row["mouth_mean"] for row in rows], [1.0, 2.0])
        self.assertEqual([row["transitions"] for row in rows], [2, 2])


if __name__ == "__main__":
    unittest.main()
