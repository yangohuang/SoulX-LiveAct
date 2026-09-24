"""Opt-in, read-only capture of generated LiveAct block latents."""

from pathlib import Path

import torch


def save_rollout_block(directory: Path, block_index: int, latent: torch.Tensor) -> Path:
    """Persist a detached CPU snapshot of one final clean latent block."""
    if block_index < 0:
        raise ValueError("block index must be nonnegative")
    if latent.ndim != 4:
        raise ValueError("latent must have four dimensions: channels, time, height, width")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"block-{block_index:04d}.pt"
    torch.save({"block_index": block_index,
                "latent": latent.detach().to("cpu", copy=True)}, path)
    return path
