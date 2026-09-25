import unittest

from audio_block_timing import block_timing, period_phases, scenario


class AudioBlockTimingTests(unittest.TestCase):
    def test_first_and_last_30_second_blocks(self):
        first = block_timing(0, audio_frames=720)
        self.assertEqual((first["output_first_frame"], first["output_last_frame"]), (0, 20))
        self.assertEqual((first["audio_first_unclamped"], first["audio_last_unclamped"]),
                         (-2, 54))
        self.assertEqual(first["streaming_audio_slice_end_exclusive"], 55)

        last = block_timing(22, audio_frames=720)
        self.assertEqual((last["output_first_frame"], last["output_last_frame"]),
                         (693, 724))
        self.assertEqual((last["audio_first_unclamped"], last["audio_last_unclamped"]),
                         (670, 726))
        self.assertEqual(last["clamped_window_positions"], 25)
        self.assertAlmostEqual(last["audio_slice_lead_from_first_output_seconds"],
                               34 / 24)

    def test_period_phases_and_aligned_candidate(self):
        self.assertEqual(period_phases(720, 3), [0, 16, 0])
        self.assertEqual(period_phases(768, 3), [0, 0, 0])
        thirty = scenario(30, 3)
        self.assertEqual(thirty["period_boundaries"][0][
            "crossing_block_output_frames_outside_2s_margin"], [])
        self.assertEqual(thirty["period_boundaries"][1][
            "crossing_block_output_frames_outside_2s_margin"], list(range(1488, 1493)))


if __name__ == "__main__":
    unittest.main()
