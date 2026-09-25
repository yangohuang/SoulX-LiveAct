import unittest
import numpy as np

from long_horizon_stability import (
    assert_compatible, boundary_indices, boundary_windows, summarize,
    third_summaries, transition_changes,
)


class LongHorizonStabilityTests(unittest.TestCase):
    def test_delayed_spike_is_not_hidden_by_immediate_transition(self):
        changes = [1.0] * 16
        changes[4] = 5.0
        changes[7] = 10.0
        window = boundary_windows(changes, [4], width=8)[0]
        self.assertEqual(window["immediate"], 5.0)
        self.assertEqual(window["peak"], 10.0)
        self.assertEqual(window["peak_offset"], 3)
        self.assertEqual(window["motion_sum"], 21.0)

    def test_summary_preserves_peak_and_total_motion(self):
        changes = [2.0] * 24
        changes[4] = 8.0
        changes[12] = 10.0
        windows = boundary_windows(changes, [4, 12], width=4)
        result = summarize(windows)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mean_immediate"], 9.0)
        self.assertEqual(result["mean_peak"], 9.0)
        self.assertEqual(result["mean_motion_sum"], 15.0)

    def test_boundary_contract_for_722_frames(self):
        boundaries = boundary_indices(722)
        self.assertEqual(len(boundaries), 22)
        self.assertEqual((boundaries[0], boundaries[-1]), (20, 692))

    def test_incomplete_window_is_rejected(self):
        with self.assertRaises(ValueError):
            boundary_windows([1.0] * 10, [5], width=8)

    def test_transition_changes_uses_adjacent_grayscale_mae(self):
        frames = np.array([[[0, 0]], [[10, 20]], [[40, 20]]], dtype=np.uint8)
        self.assertEqual(transition_changes(frames), [15.0, 15.0])

    def test_video_mismatch_rejected_before_comparison(self):
        baseline = {"width": 416, "height": 720, "frames": 722,
                    "fps": "24/1"}
        candidate = {**baseline, "frames": 721}
        with self.assertRaisesRegex(ValueError, "frames"):
            assert_compatible(baseline, candidate)

    def test_short_video_keeps_empty_horizon_thirds_explicit(self):
        windows = boundary_windows([1.0] * 40, [20], width=8)
        thirds = third_summaries(windows)
        self.assertEqual(thirds["early"]["count"], 1)
        self.assertIsNone(thirds["middle"])
        self.assertIsNone(thirds["late"])


if __name__ == "__main__":
    unittest.main()
