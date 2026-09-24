"""Small, default-off diagnostics for LiveAct's generated block boundaries."""

import torch


def block_start_frame(block_index: int, first_block_frames: int = 21,
                      later_block_frames: int = 32) -> int:
    """Map a zero-based generated block to its first decoded frame index."""
    if block_index < 0:
        raise ValueError("block index must be nonnegative")
    return 0 if block_index == 0 else first_block_frames + (block_index - 1) * later_block_frames


def capture_latent_pair(previous: torch.Tensor, current: torch.Tensor) -> dict:
    """Detach and copy the two latents on either side of a block boundary."""
    if previous.ndim != 4 or current.ndim != 4 or previous.shape[1] < 2 or current.shape[1] < 2:
        raise ValueError("both chunks must contain at least two latents")
    if previous.shape[0] != current.shape[0] or previous.shape[2:] != current.shape[2:]:
        raise ValueError("chunk channel and spatial shape must match")
    return {"previous_last_two": previous[:, -2:].detach().to("cpu", copy=True),
            "current_first_two": current[:, :2].detach().to("cpu", copy=True)}


def capture_final_context(previous: torch.Tensor, current: torch.Tensor) -> dict:
    """Keep enough clean latents to replay the target block's VAE boundary."""
    capture_latent_pair(previous, current)
    if previous.shape[1] < 5:
        raise ValueError("previous chunk must contain five latents")
    return {"previous_last_five": previous[:, -5:].detach().to("cpu", copy=True),
            "current_all": current.detach().to("cpu", copy=True)}


def latent_transition_metrics(previous: torch.Tensor, current: torch.Tensor) -> dict:
    """Measure adjacent clean-latent changes around a generated boundary."""
    if previous.ndim != 4 or current.ndim != 4 or previous.shape[1] < 2 or current.shape[1] < 2:
        raise ValueError("both chunks must contain at least two latents")
    if previous.shape[0] != current.shape[0] or previous.shape[2:] != current.shape[2:]:
        raise ValueError("chunk channel and spatial shape must match")

    def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
        diff = a.float() - b.float()
        return float(torch.sqrt(torch.mean(diff.square())))

    return {"previous_within_rmse": rmse(previous[:, -2], previous[:, -1]),
            "boundary_rmse": rmse(previous[:, -1], current[:, 0]),
            "current_within_rmse": rmse(current[:, 0], current[:, 1])}
