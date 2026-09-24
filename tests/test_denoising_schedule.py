import unittest

from denoising_schedule import denoising_schedule


class DenoisingScheduleTests(unittest.TestCase):
    def test_three_steps_preserve_baseline(self):
        self.assertEqual(denoising_schedule(3), (
            (1000.0, 937.5, 833.33333333, 0.0),
            (True, False, False),
        ))


    def test_two_steps_keep_initialization_and_final_audio_conditioning(self):
        self.assertEqual(denoising_schedule(2), (
            (1000.0, 833.33333333, 0.0),
            (True, False),
        ))


    def test_unsupported_steps_are_rejected(self):
        for steps in (0, 1, 4):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "2 or 3"):
                denoising_schedule(steps)


if __name__ == "__main__":
    unittest.main()
