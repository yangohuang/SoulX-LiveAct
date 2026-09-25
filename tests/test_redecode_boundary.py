import unittest

import torch

from redecode_boundary import (assemble_decode_latents, decoded_boundary_changes,
                               decoded_output_uint8)


class RedecodeBoundaryTests(unittest.TestCase):
    def test_uses_last_three_history_latents_before_current(self):
        previous = torch.arange(5, dtype=torch.bfloat16).reshape(1, 5, 1, 1)
        current = torch.arange(8, 16, dtype=torch.bfloat16).reshape(1, 8, 1, 1)
        latents = assemble_decode_latents(previous, current)
        torch.testing.assert_close(latents.flatten(),
                                   torch.tensor([2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15],
                                                dtype=torch.bfloat16))

    def test_rejects_unaligned_latent_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            assemble_decode_latents(torch.zeros(1, 5, 1, 1),
                                    torch.zeros(2, 8, 1, 1))

    def test_reports_local_decoder_seam_and_later_changes_separately(self):
        values = torch.tensor([0., 1., 3., 6., 10.]).reshape(1, 5, 1, 1)
        video = values.repeat(3, 1, 1, 1)
        result = decoded_boundary_changes(video, first_new=1, width=3)
        self.assertEqual(result["local_seam_mae"], 127.5)
        self.assertEqual(result["first_three_changes"], [127.5, 255.0, 382.5])

    def test_maps_decoder_range_to_mp4_rgb_byte_range(self):
        video = torch.tensor([-1., 0., 1.]).reshape(1, 3, 1, 1).repeat(3, 1, 1, 1)
        pixels = decoded_output_uint8(video, first_new=1, count=2)
        self.assertEqual(pixels.shape, (2, 1, 1, 3))
        self.assertEqual(pixels[:, 0, 0, 0].tolist(), [127, 255])


if __name__ == "__main__":
    unittest.main()
