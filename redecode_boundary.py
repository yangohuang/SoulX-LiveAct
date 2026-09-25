"""Replay one captured LiveAct block with different VAE history lengths."""

import argparse
from pathlib import Path

import torch


def assemble_decode_latents(previous: torch.Tensor, current: torch.Tensor,
                            history_count: int = 3) -> torch.Tensor:
    """Build the same causal latent input as LiveAct's later-block decoder."""
    if (previous.ndim != 4 or current.ndim != 4 or history_count < 1 or
            previous.shape[1] < history_count or current.shape[1] < 1 or
            previous.shape[0] != current.shape[0] or
            previous.shape[2:] != current.shape[2:]):
        raise ValueError("history and current latents need matching channel and spatial shape")
    return torch.cat([previous[:, -history_count:], current], dim=1)


def decoded_boundary_changes(video: torch.Tensor, first_new: int = 9,
                             width: int = 8) -> dict:
    """Score the local VAE context seam and following RGB frame changes."""
    if (video.ndim != 4 or video.shape[0] != 3 or first_new < 1 or width < 1 or
            first_new + width > video.shape[1]):
        raise ValueError("expected an RGB video with a complete boundary window")
    changes = (video[:, first_new:first_new + width] -
               video[:, first_new - 1:first_new + width - 1]).abs()
    values = (changes.float().mean(dim=(0, 2, 3)) * 127.5).tolist()
    return {"local_seam_mae": values[0],
            "first_three_changes": values[:3],
            "first_eight_peak": max(values),
            "first_eight_sum": sum(values)}


def decoded_output_uint8(video: torch.Tensor, first_new: int = 9,
                         count: int = 8) -> torch.Tensor:
    """Convert generated VAE frames to RGB bytes for source-video comparison."""
    if (video.ndim != 4 or video.shape[0] != 3 or first_new < 0 or count < 1 or
            first_new + count > video.shape[1]):
        raise ValueError("expected an RGB video with enough generated frames")
    return ((video[:, first_new:first_new + count].float() + 1.0) * 127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).contiguous()


def main(snapshot_path: Path, vae_path: Path) -> None:
    # Delay LightX2V import so offline analysis of the .pt file needs only torch.
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    previous = snapshot["previous_last_five"]
    current = snapshot["current_all"]
    vae = LightVAE(vae_path=str(vae_path), dtype=torch.bfloat16, device=0,
                   use_lightvae=False, parallel=False)
    vae.model.eval()

    for count in (3, 4, 5):
        latent = assemble_decode_latents(previous, current, count).to("cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video = vae.decode(latent.squeeze(0))
        first_new = (count - 1) * 4 + 1
        before_to_first = float((video[:, :, first_new] - video[:, :, first_new - 1]).abs().float().mean() * 127.5)
        first_to_second = float((video[:, :, first_new + 1] - video[:, :, first_new]).abs().float().mean() * 127.5)
        print(f"context={count} first_new={first_new} "
              f"before_to_first_rgb_mae={before_to_first:.3f} "
              f"first_to_second_rgb_mae={first_to_second:.3f}", flush=True)
        del latent, video
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument("vae_path", type=Path)
    args = parser.parse_args()
    main(args.snapshot_path, args.vae_path)
