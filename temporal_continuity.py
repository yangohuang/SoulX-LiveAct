"""Experimental boundary conditioning for chunked latent generation."""

import torch
import torch.nn.functional as F


def should_anchor_step(step_index: int, step_count: int, strength: float,
                       skip_last: bool = False) -> bool:
    """Choose an experimental anchor schedule without changing the default."""
    if step_count < 1 or not 0 <= step_index < step_count:
        raise ValueError("denoising step index must be within the step count")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("motion anchor strength must be between 0 and 1")
    return strength > 0.0 and not (skip_last and step_index == step_count - 1)


def anchor_chunk_start(clean_latent: torch.Tensor, previous_latent: torch.Tensor,
                       strength: float, lowpass_kernel: int = 0,
                       mode: str = "hold") -> torch.Tensor:
    """Blend the first two clean latents toward a previous-state trajectory.

    The first new latent receives ``strength``; the second receives one third
    of it. ``hold`` targets the last prior latent; ``velocity`` extrapolates
    the last two prior latents. The remaining latents stay untouched.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("motion anchor strength must be between 0 and 1")
    if lowpass_kernel not in (0, 3, 5, 9):
        raise ValueError("motion anchor lowpass kernel must be 0, 3, 5, or 9")
    if mode not in ("hold", "velocity"):
        raise ValueError("motion anchor mode must be hold or velocity")
    if (clean_latent.ndim != 4 or previous_latent.ndim != 4 or
            clean_latent.shape[0] != previous_latent.shape[0] or
            clean_latent.shape[2:] != previous_latent.shape[2:] or
            clean_latent.shape[1] < 1 or previous_latent.shape[1] < 1):
        raise ValueError("chunk latents must have matching channels and spatial shape")
    if strength == 0.0:
        return clean_latent
    if mode == "velocity" and previous_latent.shape[1] < 2:
        raise ValueError("velocity mode requires two previous latents")

    anchored = clean_latent.clone()
    last = previous_latent[:, -1:]
    velocity = last - previous_latent[:, -2:-1] if mode == "velocity" else None

    def blend(source: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        if lowpass_kernel == 0:
            return torch.lerp(source, target, weight)
        correction = F.avg_pool2d(target - source, lowpass_kernel, stride=1,
                                  padding=lowpass_kernel // 2, count_include_pad=False)
        return source + weight * correction

    anchored[:, :1] = blend(anchored[:, :1], last + velocity if velocity is not None else last,
                            strength)
    if anchored.shape[1] > 1:
        target = last + 2 * velocity if velocity is not None else last
        anchored[:, 1:2] = blend(anchored[:, 1:2], target, strength / 3.0)
    return anchored
