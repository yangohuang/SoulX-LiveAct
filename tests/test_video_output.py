import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio
import torch
from diffusers.utils import export_to_video

from video_output import VideoBlockWriter, append_video_block


def decoded_rgb(path):
    return subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        check=True, capture_output=True,
    ).stdout


class VideoOutputTests(unittest.TestCase):
    def test_writer_finalizes_video_when_generation_fails(self):
        block = torch.zeros(1, 3, 2, 16, 16, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'partial.mp4'
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                with VideoBlockWriter(output, fps=24) as writer:
                    writer.append(block)
                    raise RuntimeError('generation failed')
            self.assertEqual(len(decoded_rgb(output)), 2 * 16 * 16 * 3)

    @unittest.skipUnless(torch.cuda.is_available(), 'LiveAct CLI import requires CUDA')
    def test_stream_output_flag_is_opt_in(self):
        from generate import _parse_args

        with patch.object(sys, 'argv', ['generate.py']):
            self.assertFalse(_parse_args().stream_video_output)
        with patch.object(sys, 'argv', ['generate.py', '--stream_video_output']):
            self.assertTrue(_parse_args().stream_video_output)

    def test_streamed_blocks_match_existing_full_video_export(self):
        values = torch.linspace(-1, 1, 3 * 4 * 16 * 16, dtype=torch.float32)
        video = values.reshape(1, 3, 4, 16, 16).to(torch.bfloat16)
        blocks = [video[:, :, :2], video[:, :, 2:]]

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = directory / 'original.mp4'
            streamed = directory / 'streamed.mp4'
            frames = (torch.cat(blocks, dim=2).permute(0, 2, 3, 4, 1)[0] + 1.0) / 2
            export_to_video(frames.float().numpy(), str(original), fps=24)

            with imageio.get_writer(str(streamed), fps=24, quality=5.0,
                                    macro_block_size=16) as writer:
                for block in blocks:
                    append_video_block(writer, block)

            expected = decoded_rgb(original)
            self.assertEqual(len(expected), 4 * 16 * 16 * 3)
            self.assertEqual(decoded_rgb(streamed), expected)


if __name__ == '__main__':
    unittest.main()
