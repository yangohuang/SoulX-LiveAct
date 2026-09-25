import unittest

from boundary_signal_analysis import correlate_signal_with_motion


class BoundarySignalAnalysisTests(unittest.TestCase):
    def test_aligns_by_decoded_transition_and_reports_rank_overlap(self):
        signal = {"blocks": [
            {"block_index": i + 1, "boundary_transition": [20 + 32 * i, 21 + 32 * i],
             "steps": [{"step_index": 0, "metrics": {"gap_mae": 3 - i}},
                       {"step_index": 2, "metrics": {"gap_mae": 3 - i,
                                                       "velocity_residual_mae": 3 - i}}]}
            for i in range(3)]}
        motion = {"boundaries": [
            {"index": 20 + 32 * i, "peak": 3 - i, "immediate": 3 - i,
             "peak_offset": 0} for i in range(3)]}
        result = correlate_signal_with_motion(signal, motion, top_k=2)
        self.assertAlmostEqual(result["gap_mae"]["spearman_peak"], 1.0)
        self.assertAlmostEqual(result["gap_mae"]["spearman_immediate"], 1.0)
        self.assertEqual(result["gap_mae"]["top_k_overlap"], [1, 2])
        self.assertEqual(result["gap_mae"]["top_blocks"], [1, 2])

    def test_rejects_misaligned_decoded_boundary(self):
        signal = {"blocks": [{"block_index": 1,
                               "boundary_transition": [19, 20],
                               "steps": [{"step_index": 2,
                                          "metrics": {"gap_mae": 1.0}}]}]}
        motion = {"boundaries": [{"index": 20, "peak": 1.0}]}
        with self.assertRaisesRegex(ValueError, "alignment"):
            correlate_signal_with_motion(signal, motion)


if __name__ == "__main__":
    unittest.main()
