"""Plot boundary movement and sampled face-proxy curves for one rollout."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--face-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion = json.loads(args.motion_json.read_text())["videos"][args.label]
    face = json.loads(args.face_json.read_text())["videos"][args.label]
    if motion["sha256"] != face["sha256"]:
        raise ValueError("motion and face results use different videos")
    fps_n, fps_d = map(int, motion["metadata"]["fps"].split("/"))
    fps = fps_n / fps_d
    times = [w["index"] / fps for w in motion["boundaries"]]
    face_times = [index / fps for index in face["frame_indices"]]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(times, [w["peak"] for w in motion["boundaries"]], ".-", label="8-transition peak")
    axes[0].plot(times, [w["immediate"] for w in motion["boundaries"]], alpha=.65, label="immediate seam")
    axes[0].set_ylabel("Grayscale MAE")
    axes[0].legend(loc="upper right")
    axes[1].plot(times, [w["motion_sum"] for w in motion["boundaries"]], ".-", color="tab:green")
    axes[1].set_ylabel("8-transition motion sum")
    axes[2].plot(face_times, face["reference_cosine"], ".-", markersize=2,
                 label="reference-face cosine")
    axes[2].plot(face_times, face["first_cosine"], alpha=.65,
                 label="first-face cosine")
    axes[2].set_ylabel("Face proxy")
    axes[2].set_xlabel("Generated time (s)")
    axes[2].legend(loc="lower left")
    duration = motion["metadata"]["frames"] / fps
    for axis in axes:
        for second in range(30, int(duration), 30):
            axis.axvline(second, linestyle="--", color="gray", linewidth=.8)
        axis.grid(alpha=.2)
    fig.suptitle(f"{args.label}: causal rollout diagnostics (pose-sensitive face proxy)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
