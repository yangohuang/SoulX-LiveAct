"""Experimental boundary conditioning for chunked latent generation."""

import torch


def anchor_chunk_start(clean_latent: torch.Tensor, previous_latent: torch.Tensor,
                       strength: float, trend: float = 0.0) -> torch.Tensor:
    """Blend the first two clean latents toward the previous chunk's trajectory.

    The first new latent receives ``strength``; the second receives one third
    of it. ``trend=0`` uses the previous final latent as before. The remaining
    latents stay untouched. This is deliberately opt-in.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("motion anchor strength must be between 0 and 1")
    if not 0.0 <= trend <= 1.0:
        raise ValueError("motion anchor trend must be between 0 and 1")
    if (clean_latent.ndim != 4 or previous_latent.ndim != 4 or
            clean_latent.shape[0] != previous_latent.shape[0] or
            clean_latent.shape[2:] != previous_latent.shape[2:] or
            clean_latent.shape[1] < 1 or previous_latent.shape[1] < 1):
        raise ValueError("chunk latents must have matching channels and spatial shape")
    if trend > 0.0 and previous_latent.shape[1] < 2:
        raise ValueError("motion anchor trend requires two previous latents")
    if strength == 0.0:
        return clean_latent

    anchored = clean_latent.clone()
    last = previous_latent[:, -1:]
    delta = trend * (last - previous_latent[:, -2:-1]) if trend else 0
    anchored[:, :1] = torch.lerp(anchored[:, :1], last + delta, strength)
    if anchored.shape[1] > 1:
        anchored[:, 1:2] = torch.lerp(anchored[:, 1:2], last + 2 * delta, strength / 3.0)
    return anchored
