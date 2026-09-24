"""Replay one captured LiveAct block with different VAE history lengths."""

import argparse
from pathlib import Path

import torch


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
        latent = torch.cat([previous[:, -count:], current], dim=1).to("cuda:0")
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
