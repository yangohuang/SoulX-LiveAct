"""Replay two captured block-16 trajectories and isolate final-anchor VAE effects."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from long_horizon_stability import video_metadata
from redecode_boundary import (assemble_decode_latents, decoded_boundary_changes,
                               decoded_output_uint8)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_source_window(path: Path, start: int = 501, count: int = 8) -> np.ndarray:
    metadata = video_metadata(path)
    width, height = metadata["width"], metadata["height"]
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-vsync", "0", "-i", str(path),
        "-vf", f"select=between(n\\,{start}\\,{start + count - 1}),format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    if len(raw) != count * width * height * 3:
        raise ValueError("source video did not decode the requested frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)


def save_sheet(rows: list[tuple[str, np.ndarray]], path: Path) -> None:
    count, thumb_w, thumb_h, label_h = 8, 208, 360, 25
    canvas = Image.new("RGB", (count * thumb_w, len(rows) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (name, pixels) in enumerate(rows):
        if pixels.shape[0] != count:
            raise ValueError("contact-sheet rows must have eight frames")
        y = row_index * (thumb_h + label_h)
        draw.text((4, y + 5), name, fill="black")
        for frame_index in range(count):
            frame = Image.fromarray(pixels[frame_index]).resize((thumb_w, thumb_h))
            canvas.paste(frame, (frame_index * thumb_w, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main(baseline_path: Path, velocity_path: Path, pre_anchor_path: Path,
         baseline_video: Path, velocity_video: Path, vae_path: Path,
         output_dir: Path) -> dict:
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

    snapshots = {
        "baseline": torch.load(baseline_path, map_location="cpu", weights_only=True),
        "velocity": torch.load(velocity_path, map_location="cpu", weights_only=True),
        "velocity_pre_anchor": torch.load(pre_anchor_path, map_location="cpu", weights_only=True),
    }
    if any(snapshot["block_index"] != 16 for snapshot in snapshots.values()):
        raise ValueError("all snapshots must be generated block 16")
    if not torch.equal(snapshots["velocity"]["previous_last_five"],
                       snapshots["velocity_pre_anchor"]["previous_last_five"]):
        raise ValueError("pre/post anchor snapshots must share exact prior latents")

    vae = LightVAE(vae_path=str(vae_path), dtype=torch.bfloat16, device=0,
                   use_lightvae=False, parallel=False)
    vae.model.eval()
    source_baseline = load_source_window(baseline_video)
    source_velocity = load_source_window(velocity_video)

    def decode(previous: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, float]:
        latent = assemble_decode_latents(previous, current).to("cuda:0", torch.bfloat16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video = vae.decode(latent.squeeze(0))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"expected RGB VAE output; got {tuple(video.shape)}")
        return video.float().cpu(), seconds

    results = {}
    pixels = {}
    variants = (
        ("baseline", "baseline", "baseline"),
        ("velocity", "velocity", "velocity"),
        ("velocity_pre_anchor", "velocity", "velocity_pre_anchor"),
        ("baseline_history_velocity_current", "baseline", "velocity"),
        ("velocity_history_baseline_current", "velocity", "baseline"),
    )
    for name, previous_arm, current_arm in variants:
        prior = snapshots[previous_arm]["previous_last_five"]
        current = snapshots[current_arm]["current_all"]
        video, seconds = decode(prior, current)
        results[name] = {"history_arm": previous_arm, "current_arm": current_arm,
                         "decode_seconds": seconds,
                         "local_decoder_changes": decoded_boundary_changes(video)}
        pixels[name] = decoded_output_uint8(video).numpy()
        del video
        if name in ("baseline", "velocity"):
            source = source_baseline if name == "baseline" else source_velocity
            difference = np.abs(source.astype(np.int16) - pixels[name].astype(np.int16))
            results[name]["source_replay_mean_rgb_mae"] = float(difference.mean())
            results[name]["source_replay_max_rgb_difference"] = int(difference.max())
        if name == "velocity" and any(results[arm]["source_replay_mean_rgb_mae"] > 8.0
                                      for arm in ("baseline", "velocity")):
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {"same_arm_validation_passed": False,
                       "reason": "same-arm VAE replay exceeds the prespecified 8/255 mean RGB gate",
                       "results": results}
            (output_dir / "block16-validation-failure.json").write_text(
                json.dumps(failure, indent=2) + "\n")
            print(json.dumps(failure, indent=2), flush=True)
            return failure

    before = snapshots["velocity_pre_anchor"]["current_all"].float()
    after = snapshots["velocity"]["current_all"].float()
    if not torch.equal(before[:, 2:], after[:, 2:]):
        raise ValueError("final anchor unexpectedly changed latents beyond the first two")
    correction = after[:, :2] - before[:, :2]
    correction_metrics = {"first_two_rmse": float(correction.square().mean().sqrt()),
                          "first_two_mae": float(correction.abs().mean()),
                          "other_latents_equal": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "block16-seven-row-replay.png"
    save_sheet([
        ("baseline MP4 frames 501-508", source_baseline),
        ("baseline VAE replay", pixels["baseline"]),
        ("velocity MP4 frames 501-508", source_velocity),
        ("velocity VAE replay", pixels["velocity"]),
        ("velocity PRE final anchor", pixels["velocity_pre_anchor"]),
        ("baseline history + velocity current", pixels["baseline_history_velocity_current"]),
        ("velocity history + baseline current", pixels["velocity_history_baseline_current"]),
    ], sheet_path)
    report = {"block_index": 16, "output_frames": [501, 508],
              "same_arm_validation_passed": all(
                  results[arm]["source_replay_mean_rgb_mae"] <= 8.0
                  for arm in ("baseline", "velocity")),
              "correction": correction_metrics,
              "videos": {arm: {"source": str(path.resolve()),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for arm, path in (("baseline", baseline_video),
                                           ("velocity", velocity_video))},
              "snapshots": {name: str(path.resolve()) for name, path in (
                  ("baseline", baseline_path), ("velocity", velocity_path),
                  ("velocity_pre_anchor", pre_anchor_path))},
              "results": results, "contact_sheet": str(sheet_path.resolve())}
    (output_dir / "block16-replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--velocity-snapshot", type=Path, required=True)
    parser.add_argument("--velocity-pre-anchor", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--velocity-video", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline_snapshot, args.velocity_snapshot, args.velocity_pre_anchor,
         args.baseline_video, args.velocity_video, args.vae, args.output_dir)
