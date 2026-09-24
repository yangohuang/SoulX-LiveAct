"""Replay a captured LiveAct boundary with a frozen-backbone latent bridge."""

import argparse
import json
from pathlib import Path

import torch

from rollout_bridge import RolloutBridge, bridge_chunk_start


def canonical_video(video: torch.Tensor) -> torch.Tensor:
    """Remove the singleton batch dimension returned by LightVAE."""
    if video.ndim == 5 and video.shape[0] == 1:
        video = video.squeeze(0)
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"expected RGB video, got shape {tuple(video.shape)}")
    return video


def transition_rgb_mae(video: torch.Tensor, first_new: int = 9) -> list[float]:
    """Adjacent decoded RGB changes after the first new output frame."""
    if video.ndim != 4 or video.shape[0] != 3 or first_new < 0 or first_new >= video.shape[1] - 1:
        raise ValueError("video must be RGB, channels-first, with two new frames")
    changes = (video[:, first_new + 1:] - video[:, first_new:-1]).abs()
    return (changes.float().mean(dim=(0, 2, 3)) * 127.5).tolist()


def save_sheet(baseline: torch.Tensor, corrected: torch.Tensor,
               first_new: int, output: Path) -> None:
    from PIL import Image, ImageDraw

    count, width, height, label_height = 9, 208, 360, 28
    canvas = Image.new("RGB", (count * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, video) in enumerate((("baseline", baseline), ("bridge", corrected))):
        for offset in range(count):
            frame = video[:, first_new + offset]
            pixels = ((frame + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            x, y = offset * width, row * (height + label_height)
            canvas.paste(Image.fromarray(pixels).resize((width, height)), (x, y + label_height))
            draw.text((x + 5, y + 5), f"{name} +{offset}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(checkpoint_path: Path, snapshot_path: Path, vae_path: Path,
         output_path: Path) -> dict:
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    previous = snapshot["previous_last_five"]
    current = snapshot["current_all"]
    model = RolloutBridge(channels=checkpoint["channels"],
                          hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        changed = bridge_chunk_start(model, previous.float(), current.float())
    correction_rmse = float((changed[:, :2] - current[:, :2].float()).square().mean().sqrt())
    vae = LightVAE(vae_path=str(vae_path), dtype=torch.bfloat16, device=0,
                   use_lightvae=False, parallel=False)
    vae.model.eval()

    def decode(chunk: torch.Tensor) -> torch.Tensor:
        latent = torch.cat([previous[:, -3:], chunk], dim=1).to("cuda:0", torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return canonical_video(vae.decode(latent.squeeze(0)).float().cpu())

    baseline_video = decode(current)
    corrected_video = decode(changed)
    first_new = 9
    baseline_changes = transition_rgb_mae(baseline_video, first_new)
    corrected_changes = transition_rgb_mae(corrected_video, first_new)
    save_sheet(baseline_video, corrected_video, first_new, output_path)
    report = {"snapshot": str(snapshot_path), "checkpoint": str(checkpoint_path),
              "contact_sheet": str(output_path), "correction_rmse": correction_rmse,
              "baseline_transitions_first12": baseline_changes[:12],
              "bridge_transitions_first12": corrected_changes[:12],
              "baseline_max_first12": max(baseline_changes[:12]),
              "bridge_max_first12": max(corrected_changes[:12])}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("vae_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.snapshot, args.vae_path, args.output)
