import argparse
import unittest

import torch

import runtime_options


class CudnnReproducibilityTests(unittest.TestCase):
    def test_opt_in_disables_benchmark_without_changing_deterministic_flag(self):
        parser = argparse.ArgumentParser()
        runtime_options.add_low_memory_arguments(parser)
        args = parser.parse_args(["--disable_cudnn_benchmark"])
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_deterministic = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            runtime_options.configure_cudnn_benchmark(args.disable_cudnn_benchmark)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.deterministic = previous_deterministic

    def test_default_preserves_existing_benchmark_policy(self):
        parser = argparse.ArgumentParser()
        runtime_options.add_low_memory_arguments(parser)
        args = parser.parse_args([])
        previous = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = True
            runtime_options.configure_cudnn_benchmark(args.disable_cudnn_benchmark)
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.benchmark = previous


if __name__ == "__main__":
    unittest.main()
