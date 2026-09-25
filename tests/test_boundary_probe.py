import unittest

import torch

from boundary_probe import (block_start_frame, capture_latent_pair,
                            capture_final_context, latent_transition_metrics,
                            pre_output_boundary_metrics, write_boundary_signal)


class BoundaryProbeTests(unittest.TestCase):
    def test_pre_output_metric_distinguishes_continuing_motion(self):
        previous = torch.tensor([[[[1.]], [[3.]]]], dtype=torch.bfloat16)
        predicted = torch.tensor([[[[5.]], [[8.]]]], dtype=torch.bfloat16)
        metrics = pre_output_boundary_metrics(previous, predicted)
        self.assertEqual({k: v.item() for k, v in metrics.items()},
                         {"gap_mae": 2.0, "previous_speed_mae": 2.0,
                          "velocity_residual_mae": 0.0})
        self.assertEqual({v.device.type for v in metrics.values()}, {"cpu"})
        with self.assertRaisesRegex(ValueError, "shape"):
            pre_output_boundary_metrics(previous, torch.zeros(2, 2, 1, 1))

    def test_signal_writer_serializes_first_and_final_steps(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request-0.json"
            blocks = [{"block_index": 1, "steps": [
                {"step_index": 0, "timestep": 1000.0,
                 "metrics": {"gap_mae": torch.tensor(2.0)}},
                {"step_index": 2, "timestep": 350.0,
                 "metrics": {"gap_mae": torch.tensor(1.5)}}]}]
            write_boundary_signal(path, blocks, seed=43, image_path="example.png",
                                  audio_path="example.wav", output_path="result.mp4",
                                  settings={"denoising_steps": 3})
            data = json.loads(path.read_text())
            self.assertEqual(data["blocks"][0]["first_output_frame"], 21)
            self.assertEqual(data["blocks"][0]["boundary_transition"], [20, 21])
            self.assertEqual(data["blocks"][0]["steps"][1]["metrics"]["gap_mae"], 1.5)
            self.assertEqual(data["seed"], 43)
            self.assertEqual(data["settings"], {"denoising_steps": 3})

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
