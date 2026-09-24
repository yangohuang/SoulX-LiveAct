"""Export decoded video blocks without concatenating the whole clip."""

def export_video_blocks(blocks, path, fps):
    """Write [1, C, T, H, W] blocks with the same conversion as export_to_video."""
    import imageio
    import numpy as np

    with imageio.get_writer(str(path), fps=fps, quality=5.0,
                            macro_block_size=16) as writer:
        for block in blocks:
            frames = (block.permute(0, 2, 3, 4, 1)[0] + 1.0) / 2
            for frame in frames:
                writer.append_data((frame.float().numpy() * 255).astype(np.uint8))
