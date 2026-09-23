import argparse
import unittest
from unittest import mock

import torch
from torch import nn

from runtime_options import add_low_memory_arguments, maybe_compile, offload_blocks


class RuntimeOptionsTests(unittest.TestCase):
    def test_existing_compile_default_is_preserved_but_pinning_is_opt_in(self):
        parser = argparse.ArgumentParser()
        add_low_memory_arguments(parser)

        args = parser.parse_args([])

        self.assertFalse(args.disable_compile)
        self.assertFalse(args.pin_block_memory)

    def test_low_memory_compile_override_and_pinning_flag_are_explicit(self):
        parser = argparse.ArgumentParser()
        add_low_memory_arguments(parser)

        args = parser.parse_args(["--disable_compile", "--pin_block_memory"])

        self.assertTrue(args.disable_compile)
        self.assertTrue(args.pin_block_memory)

    def test_maybe_compile_returns_module_unchanged_when_disabled(self):
        module = nn.Linear(2, 2)
        compiler = mock.Mock()

        result = maybe_compile(module, enabled=False, compiler=compiler)

        self.assertIs(result, module)
        compiler.assert_not_called()

    def test_offload_blocks_only_pins_when_requested(self):
        blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        pin_module = mock.Mock()

        offload_blocks(blocks, "cpu", False, pin_module)
        pin_module.assert_not_called()

        offload_blocks(blocks, "cpu", True, pin_module)
        self.assertEqual(pin_module.call_count, 2)
        self.assertTrue(all(block.weight.device.type == "cpu" for block in blocks))


if __name__ == "__main__":
    unittest.main()
