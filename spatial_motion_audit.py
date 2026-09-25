"""CPU-only fixed-region image-change and optical-flow audit for LiveAct videos."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from long_horizon_stability import assert_compatible, boundary_indices, decode_grayscale, video_metadata


HEIGHT, WIDTH = 180, 104
REGION_BOXES = {
    "whole": [[0, 0, WIDTH, HEIGHT]],
    "mouth": [[36, 37, 68, 64]],
    "upper_head": [[25, 8, 79, 82]],
    "lower_body": [[0, 105, WIDTH, HEIGHT]],
    "background": [[0, 0, 20, 70], [84, 0, WIDTH, 70]],
}


def region_masks() -> dict[str, np.ndarray]:
    """Return fixed scene-space masks; these are not semantic segmentations."""
    masks = {name: np.zeros((HEIGHT, WIDTH), dtype=bool) for name in REGION_BOXES}
    for name, boxes in REGION_BOXES.items():
        for x0, y0, x1, y1 in boxes:
            masks[name][y0:y1, x0:x1] = True
    return masks


REGION_MASKS = region_masks()


def measure_pair(previous: np.ndarray, current: np.ndarray) -> dict[str, dict[str, float]]:
    """Measure adjacent 104×180 gray frames in fixed spatial masks."""
    import cv2

    if previous.shape != (HEIGHT, WIDTH) or current.shape != (HEIGHT, WIDTH):
        raise ValueError("both frames must be 104x180 grayscale")
    if previous.dtype != np.uint8 or current.dtype != np.uint8:
        raise ValueError("both frames must be uint8")
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 11, 2, 5, 1.1, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    return {name: {"mae": float(difference[mask].mean()),
                   "flow": float(magnitude[mask].mean())}
            for name, mask in REGION_MASKS.items()}


def boundary_sums(changes: list[float], frame_count: int) -> list[dict]:
    """Sum the existing eight-transition windows at complete chunk boundaries."""
    if len(changes) != frame_count - 1:
        raise ValueError("one change is required for every adjacent frame pair")
    return [{"index": index, "sum": float(sum(changes[index:index + 8]))}
            for index in boundary_indices(frame_count)]


def analyze_video(path: Path) -> dict:
    metadata = video_metadata(path)
    frames = decode_grayscale(path, metadata["frames"], width=WIDTH, height=HEIGHT)
    if len(frames) < 29:
        raise ValueError("video has no complete chunk boundary")
    timelines = {name: {metric: [] for metric in ("mae", "flow")}
                 for name in REGION_MASKS}
    for previous, current in zip(frames[:-1], frames[1:]):
        measured = measure_pair(previous, current)
        for name in REGION_MASKS:
            for metric in ("mae", "flow"):
                timelines[name][metric].append(measured[name][metric])
    summaries = {}
    for name, metrics in timelines.items():
        summaries[name] = {}
        for metric, changes in metrics.items():
            windows = boundary_sums(changes, metadata["frames"])
            summaries[name][metric] = {
                "all_transition_mean": float(np.mean(changes)),
                "boundary_mean_sum": float(np.mean([row["sum"] for row in windows])),
                "boundary_sums": windows,
            }
    return {"source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": metadata, "timelines": timelines, "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({label for label, _ in args.video}) != len(args.video):
        parser.error("video labels must be unique")
    videos = {label: analyze_video(Path(path)) for label, path in args.video}
    reference = next(iter(videos.values()))["metadata"]
    for video in videos.values():
        assert_compatible(reference, video["metadata"])
    payload = {
        "contract": "fixed 104x180 grayscale regions; adjacent-frame MAE and "
                    "Farneback optical-flow magnitude; complete eight-transition "
                    "windows at indices 20+32k",
        "regions": {name: {"boxes_xyxy": REGION_BOXES[name],
                            "pixels": int(mask.sum())}
                    for name, mask in REGION_MASKS.items()},
        "farneback": {"pyr_scale": 0.5, "levels": 2, "winsize": 11,
                       "iterations": 2, "poly_n": 5, "poly_sigma": 1.1, "flags": 0},
        "videos": videos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({label: {name: {metric: value["boundary_mean_sum"]
                                           for metric, value in summary.items()}
                                  for name, summary in video["summaries"].items()}
                      for label, video in videos.items()}, indent=2))


if __name__ == "__main__":
    main()
