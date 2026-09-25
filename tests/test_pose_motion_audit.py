import unittest

from pose_motion_audit import displacements


class PoseMotionAuditTests(unittest.TestCase):
    def test_displacements_exclude_low_visibility_and_out_of_frame_points(self):
        landmarks = [
            {"x": 0.1, "y": 0.1, "visibility": 1.0},
            {"x": 0.4, "y": 0.5, "visibility": 0.9},
            {"x": 0.6, "y": 0.5, "visibility": 0.4},
            {"x": 1.1, "y": 0.5, "visibility": 0.9},
            {"x": 0.7, "y": 0.5, "visibility": 0.9},
        ]

        result = displacements(landmarks)

        self.assertAlmostEqual(result[0], 0.5)
        self.assertEqual(result[1:], [None, None, None])

    def test_missing_detections_break_trajectory(self):
        landmarks = [{"x": 0.1, "y": 0.2, "visibility": 1.0}, None,
                     {"x": 0.2, "y": 0.2, "visibility": 1.0}]
        self.assertEqual(displacements(landmarks), [None, None])


if __name__ == "__main__":
    unittest.main()
