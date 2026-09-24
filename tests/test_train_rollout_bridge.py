import tempfile
import unittest
from pathlib import Path

import torch

from rollout_bridge import RolloutBridge
from rollout_capture import save_rollout_block
from train_rollout_bridge import draw_batch, evaluate_bridge, load_blocks


class TrainRolloutBridgeTests(unittest.TestCase):
    def test_loads_only_captured_blocks_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_rollout_block(directory, 1, torch.ones(2, 8, 3, 3))
            save_rollout_block(directory, 0, torch.zeros(2, 6, 3, 3))
            (directory / "notes.txt").write_text("ignored")

            blocks = load_blocks([directory])

        self.assertEqual(len(blocks), 2)
        self.assertEqual(tuple(blocks[0].shape), (2, 6, 3, 3))
        self.assertEqual(float(blocks[1].mean()), 1.0)

    def test_fixed_seed_batches_repeat_and_noop_baseline_matches_model(self):
        block = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        block = block.expand(1, 8, 8, 8).clone()
        first = draw_batch([block], batch_size=3, seed=17)
        second = draw_batch([block], batch_size=3, seed=17)
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected)

        model = RolloutBridge(channels=1, hidden=4)
        score = evaluate_bridge(model, [block], sample_count=8, seed=3)
        self.assertAlmostEqual(score["model_l1"], score["uncorrected_l1"], places=6)
        self.assertAlmostEqual(score["identity_drift_l1"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
