import unittest

from horizon_trend import analyze_horizon


class HorizonTrendTests(unittest.TestCase):
    def test_fixed_time_segments_do_not_hide_late_motion_collapse(self):
        # The late peaks are larger, but the generated motion is much smaller.
        windows = [
            {"index": 20, "peak": 4.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 340, "peak": 5.0, "motion_sum": 32.0, "immediate": 3.0},
            {"index": 760, "peak": 5.5, "motion_sum": 31.0, "immediate": 3.0},
            {"index": 1440, "peak": 5.0, "motion_sum": 29.0, "immediate": 2.0},
            {"index": 1480, "peak": 7.0, "motion_sum": 20.0, "immediate": 4.0},
            {"index": 1800, "peak": 8.0, "motion_sum": 22.0, "immediate": 4.0},
        ]
        result = analyze_horizon(windows, fps=24, segment_seconds=30,
                                 segment_count=3)
        self.assertEqual([row["count"] for row in result["segments"]], [2, 1, 3])
        self.assertFalse(result["deterioration_signal"])
        self.assertFalse(result["motion_retained"])

    def test_peak_increase_requires_motion_retention(self):
        windows = [
            {"index": 20, "peak": 4.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 340, "peak": 4.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 760, "peak": 4.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 1440, "peak": 6.0, "motion_sum": 29.0, "immediate": 3.0},
            {"index": 1480, "peak": 7.0, "motion_sum": 29.0, "immediate": 3.0},
        ]
        result = analyze_horizon(windows, fps=24, segment_seconds=30,
                                 segment_count=3)
        self.assertTrue(result["deterioration_signal"])
        self.assertTrue(result["motion_retained"])

    def test_isolated_tail_events_do_not_override_lower_typical_peak(self):
        windows = [
            {"index": 20, "peak": 6.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 340, "peak": 6.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 760, "peak": 6.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 1440, "peak": 5.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 1480, "peak": 7.0, "motion_sum": 30.0, "immediate": 2.0},
            {"index": 1800, "peak": 5.0, "motion_sum": 30.0, "immediate": 2.0},
        ]
        result = analyze_horizon(windows, fps=24, segment_seconds=30,
                                 segment_count=3)
        self.assertEqual(result["segments"][-1]["severe_count"], 1)
        self.assertFalse(result["deterioration_signal"])


if __name__ == "__main__":
    unittest.main()
