import unittest

import torch

from boundary_probe import (block_start_frame, capture_latent_pair,
                            capture_final_context, latent_transition_metrics)


class BoundaryProbeTests(unittest.TestCase):
    def test_frame_mapping_after_initial_six_latent_block(self):
        self.assertEqual(block_start_frame(0), 0)
        self.assertEqual(block_start_frame(1), 21)
        self.assertEqual(block_start_frame(12), 373)

    def test_capture_copies_only_two_boundary_latents(self):
        previous = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
        current = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)

        snapshot = capture_latent_pair(previous, current)
        previous[:, -1] = 99
        current[:, 0] = 99

        torch.testing.assert_close(snapshot["previous_last_two"].flatten(), torch.tensor([3., 4.]))
        torch.testing.assert_close(snapshot["current_first_two"].flatten(), torch.tensor([0., 1.]))
        self.assertEqual(snapshot["previous_last_two"].device.type, "cpu")

    def test_capture_rejects_short_or_incompatible_chunks(self):
        with self.assertRaisesRegex(ValueError, "two"):
            capture_latent_pair(torch.zeros(1, 1, 1, 1), torch.zeros(1, 2, 1, 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            capture_latent_pair(torch.zeros(1, 2, 1, 1), torch.zeros(2, 2, 1, 1))

    def test_final_context_keeps_five_previous_and_entire_current_chunk(self):
        previous = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)
        current = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)

        snapshot = capture_final_context(previous, current)

        torch.testing.assert_close(snapshot["previous_last_five"].flatten(),
                                   torch.tensor([1., 2., 3., 4., 5.]))
        torch.testing.assert_close(snapshot["current_all"].flatten(),
                                   torch.arange(8, dtype=torch.float32))

    def test_transition_metrics_compare_adjacent_latents(self):
        previous = torch.tensor([[[[0.]], [[1.]]]])
        current = torch.tensor([[[[4.]], [[5.]]]])

        metrics = latent_transition_metrics(previous, current)

        self.assertEqual(metrics["previous_within_rmse"], 1.0)
        self.assertEqual(metrics["boundary_rmse"], 3.0)
        self.assertEqual(metrics["current_within_rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
