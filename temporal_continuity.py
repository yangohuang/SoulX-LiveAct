"""Experimental boundary conditioning for chunked latent generation."""

import torch


def anchor_chunk_start(clean_latent: torch.Tensor, previous_latent: torch.Tensor,
                       strength: float) -> torch.Tensor:
    """Blend the first two clean latents toward the previous chunk's last latent.

    The first new latent receives ``strength``; the second receives one third
    of it. The remaining latents stay untouched. This is deliberately opt-in.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("motion anchor strength must be between 0 and 1")
    if (clean_latent.ndim != 4 or previous_latent.ndim != 4 or
            clean_latent.shape[0] != previous_latent.shape[0] or
            clean_latent.shape[2:] != previous_latent.shape[2:] or
            clean_latent.shape[1] < 1 or previous_latent.shape[1] < 1):
        raise ValueError("chunk latents must have matching channels and spatial shape")
    if strength == 0.0:
        return clean_latent

    anchored = clean_latent.clone()
    last = previous_latent[:, -1:]
    anchored[:, :1] = torch.lerp(anchored[:, :1], last, strength)
    if anchored.shape[1] > 1:
        anchored[:, 1:2] = torch.lerp(anchored[:, 1:2], last, strength / 3.0)
    return anchored
