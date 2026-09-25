"""Plot original and controlled request-end decoded-pixel divergence."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.artifact_dir
    codec = json.loads((root / "codec-only-comparison.json").read_text())
    five = json.loads((root / "short-pair-mp4-comparison.json").read_text())
    ten = json.loads((root / "clamp-pair-mp4-comparison.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    panels = [
        (axes[0], [(codec["original"]["per_frame_mae_0_255"], "original 30/90"),
                   (codec["codec_only"]["per_frame_mae_0_255"], "same-source H.264 control")],
         693, "30s final model block"),
        (axes[1], [(five["frame_rgb_mae_0_255"], "5/10 MP4")],
         None, "all 117 pre-encoder frames equal"),
        (axes[2], [(ten["frame_rgb_mae_0_255"], "10/20 MP4")],
         213, "first differing pre-encoder frame"),
    ]
    for axis, lines, marker, title in panels:
        for values, label in lines:
            axis.plot(range(len(values)), values, label=label, linewidth=1.1)
        if marker is not None:
            axis.axvline(marker, color="black", linestyle="--", alpha=0.7)
        axis.set_title(title)
        axis.set_ylabel("RGB MAE / 255")
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Common decoded frame index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
