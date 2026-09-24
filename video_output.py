"""Append decoded LiveAct blocks to an imageio video writer."""

import imageio
import numpy as np


def append_video_block(writer, block):
    """Write a [1, C, T, H, W] BF16 block with the legacy export conversion."""
    frames = (block.cpu().permute(0, 2, 3, 4, 1)[0] + 1.0) / 2
    for frame in frames:
        writer.append_data((frame.float().numpy() * 255).astype(np.uint8))


class VideoBlockWriter:
    """Own one MP4 writer and close it even when block generation fails."""

    def __init__(self, path, fps):
        self._writer = imageio.get_writer(str(path), fps=fps, quality=5.0,
                                          macro_block_size=16)

    def append(self, block):
        append_video_block(self._writer, block)

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
