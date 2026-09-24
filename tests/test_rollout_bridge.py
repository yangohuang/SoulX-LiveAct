import unittest

import torch

from rollout_bridge import RolloutBridge, bridge_chunk_start, make_pseudo_boundary


class RolloutBridgeTests(unittest.TestCase):
    def test_pseudo_boundary_corrupts_only_first_two_future_latents(self):
        block = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        history, current, target = make_pseudo_boundary(block, seam=2, mix=0.5,
                                                         donor_shift=3)
        torch.testing.assert_close(history.flatten(), torch.tensor([0., 1.]))
        torch.testing.assert_close(target.flatten(), torch.tensor([2., 3.]))
        torch.testing.assert_close(current.flatten(), torch.tensor([3.5, 4.5, 4.]))
        torch.testing.assert_close(block.flatten(), torch.arange(8, dtype=torch.float32))

    def test_zero_mix_preserves_clean_future(self):
        block = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        _, current, target = make_pseudo_boundary(block, seam=3, mix=0.0,
                                                  donor_shift=3)
        torch.testing.assert_close(current[:, :2], target)

    def test_model_is_initially_noop_and_leaves_third_future_latent_alone(self):
        torch.manual_seed(2)
        history = torch.randn(2, 4, 2, 8, 8)
        current = torch.randn(2, 4, 3, 8, 8)
        model = RolloutBridge(channels=4, hidden=8)

        actual = model(history, current)

        torch.testing.assert_close(actual, current)
        self.assertIsNot(actual, current)

    def test_gradients_flow_only_through_bridge_parameters(self):
        model = RolloutBridge(channels=4, hidden=8)
        history = torch.randn(2, 4, 2, 8, 8)
        current = torch.randn(2, 4, 3, 8, 8)
        target = torch.randn(2, 4, 2, 8, 8)

        (model(history, current)[:, :, :2] - target).square().mean().backward()

        self.assertIsNone(history.grad)
        self.assertIsNone(current.grad)
        self.assertIsNotNone(model.output.weight.grad)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.0)

    def test_chunk_application_preserves_later_latents_and_inputs(self):
        previous = torch.randn(4, 8, 8, 8)
        current = torch.randn(4, 8, 8, 8)
        original = current.clone()
        model = RolloutBridge(channels=4, hidden=8)

        corrected = bridge_chunk_start(model, previous, current)

        self.assertEqual(tuple(corrected.shape), tuple(current.shape))
        torch.testing.assert_close(corrected[:, 2:], current[:, 2:])
        torch.testing.assert_close(current, original)


if __name__ == "__main__":
    unittest.main()
