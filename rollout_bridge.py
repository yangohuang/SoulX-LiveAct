"""A small residual bridge trained on LiveAct's own generated latents.

This is a frozen-backbone research module, not Self-Replay Forcing.
"""

import torch
from torch import nn


def make_pseudo_boundary(block: torch.Tensor, seam: int, mix: float,
                         donor_shift: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return history, corrupted future, and clean target from one block.

    ``block`` has (channels, time, height, width). Only the first two future
    latents are corrupted; the third remains a clean look-ahead cue.
    """
    if block.ndim != 4:
        raise ValueError("block must have four dimensions")
    if not 2 <= seam <= block.shape[1] - 3:
        raise ValueError("seam must leave two history and three future latents")
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if donor_shift % block.shape[1] == 0:
        raise ValueError("donor_shift must select different latents")

    history = block[:, seam - 2:seam].clone()
    target = block[:, seam:seam + 2].clone()
    current = block[:, seam:seam + 3].clone()
    indices = [(seam + offset + donor_shift) % block.shape[1] for offset in range(2)]
    donor = block[:, indices]
    current[:, :2] = torch.lerp(target, donor, mix)
    return history, current, target


class RolloutBridge(nn.Module):
    """Predict a residual for the first two future latents from local context."""

    def __init__(self, channels: int = 16, hidden: int = 64):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(5 * channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.output = nn.Conv2d(hidden, 2 * channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or current.ndim != 5:
            raise ValueError("history and current must have batch, channel, time, height, width")
        batch, channels, _, height, width = history.shape
        if (channels != self.channels or history.shape[2] != 2 or
                current.shape != (batch, channels, 3, height, width)):
            raise ValueError("history and current shapes do not match bridge configuration")

        combined = torch.cat([history, current], dim=2)
        features = combined.permute(0, 2, 1, 3, 4).reshape(batch, 5 * channels, height, width)
        delta = self.output(self.net(features))
        delta = delta.reshape(batch, 2, channels, height, width).permute(0, 2, 1, 3, 4)
        return torch.cat([current[:, :, :2] + delta, current[:, :, 2:]], dim=2)


def bridge_chunk_start(model: RolloutBridge, previous: torch.Tensor,
                       current: torch.Tensor) -> torch.Tensor:
    """Correct only the first two latents of an entire new chunk."""
    if previous.ndim != 4 or current.ndim != 4 or previous.shape[1] < 2 or current.shape[1] < 3:
        raise ValueError("previous needs two latents and current needs three")
    if previous.shape[0] != current.shape[0] or previous.shape[2:] != current.shape[2:]:
        raise ValueError("chunk channel and spatial shape must match")
    history = previous[:, -2:].unsqueeze(0)
    future = current[:, :3].unsqueeze(0)
    corrected = model(history, future).squeeze(0)
    return torch.cat([corrected, current[:, 3:]], dim=1)
