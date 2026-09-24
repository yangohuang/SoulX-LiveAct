import unittest

import torch

from evaluate_rollout_boundary import canonical_video, transition_rgb_mae


class EvaluateRolloutBoundaryTests(unittest.TestCase):
    def test_accepts_single_batch_vae_output(self):
        video = torch.zeros(1, 3, 12, 2, 2)
        self.assertEqual(tuple(canonical_video(video).shape), (3, 12, 2, 2))

    def test_transition_scores_start_at_first_new_frame(self):
        video = torch.zeros(3, 12, 2, 2)
        video[:, 9] = 0.2
        video[:, 10] = 0.4
        video[:, 11] = 0.5

        scores = transition_rgb_mae(video, first_new=9)

        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 25.5, places=4)
        self.assertAlmostEqual(scores[1], 12.75, places=4)


if __name__ == "__main__":
    unittest.main()
