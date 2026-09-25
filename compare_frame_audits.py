"""Compare two opt-in LiveAct frame audits in their shared video prefix."""

import argparse
import json
from pathlib import Path

from frame_audit import first_divergence


FIELDS = ("audio_window_sha256", "initial_noise_sha256",
          "final_latent_sha256", "decoded_block_sha256")


def compare(short: dict, long: dict, short_features: dict, long_features: dict) -> dict:
    if short["seed"] != long["seed"] or short["fps"] != long["fps"]:
        raise ValueError("seed and FPS must match")
    if short_features["source_embedding_sha256"] != long_features["source_embedding_sha256"]:
        raise ValueError("canonical source feature tensor must match")
    shared_blocks = min(len(short["blocks"]), len(long["blocks"]))
    shared_frames = min(len(short["pre_encoder_frame_sha256"]),
                        len(long["pre_encoder_frame_sha256"]))
    return {
        "seed": short["seed"], "fps": short["fps"],
        "source_embedding_sha256": short_features["source_embedding_sha256"],
        "shared_blocks": shared_blocks, "shared_pre_encoder_frames": shared_frames,
        "first_different_block_by_field": {
            field: first_divergence(
                [block[field] for block in short["blocks"][:shared_blocks]],
                [block[field] for block in long["blocks"][:shared_blocks]])
            for field in FIELDS},
        "first_different_pre_encoder_frame": first_divergence(
            short["pre_encoder_frame_sha256"][:shared_frames],
            long["pre_encoder_frame_sha256"][:shared_frames]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-audit", type=Path, required=True)
    parser.add_argument("--long-audit", type=Path, required=True)
    parser.add_argument("--short-features", type=Path, required=True)
    parser.add_argument("--long-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = compare(*(json.loads(path.read_text()) for path in (
        args.short_audit, args.long_audit, args.short_features, args.long_features)))
    data["inputs"] = {name: str(getattr(args, name)) for name in
                      ("short_audit", "long_audit", "short_features", "long_features")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
