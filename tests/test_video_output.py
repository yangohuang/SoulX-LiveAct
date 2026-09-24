import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from video_output import export_video_blocks


def decoded_rgb(path):
    return subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        check=True, capture_output=True,
    ).stdout


class VideoOutputTests(unittest.TestCase):
    def test_blockwise_export_matches_existing_full_video_export(self):
        values = torch.linspace(-1, 1, 3 * 4 * 16 * 16, dtype=torch.float32)
        video = values.reshape(1, 3, 4, 16, 16).to(torch.bfloat16)
        blocks = [video[:, :, :2], video[:, :, 2:]]

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = directory / 'original.mp4'
            blockwise = directory / 'blockwise.mp4'
            frames = (torch.cat(blocks, dim=2).permute(0, 2, 3, 4, 1)[0] + 1.0) / 2
            export_to_video(frames.float().numpy(), str(original), fps=24)
            export_video_blocks(blocks, blockwise, fps=24)

            expected = decoded_rgb(original)
            self.assertEqual(len(expected), 4 * 16 * 16 * 3)
            self.assertEqual(decoded_rgb(blockwise), expected)


if __name__ == '__main__':
    unittest.main()
