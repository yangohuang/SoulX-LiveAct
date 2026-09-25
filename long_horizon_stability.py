"""CPU-only transition diagnostics for fixed-chunk LiveAct rollouts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean

import numpy as np


def boundary_indices(frame_count: int, first: int = 20, stride: int = 32,
                     width: int = 8) -> list[int]:
    """Return adjacent-frame transition indices at complete chunk windows."""
    if frame_count < 2 or first < 0 or stride < 1 or width < 1:
        raise ValueError("invalid frame count or chunk layout")
    return list(range(first, frame_count - width, stride))


def boundary_windows(changes: list[float], boundaries: list[int],
                     width: int = 8) -> list[dict]:
    """Measure both the immediate seam and delayed motion within each window."""
    if width < 1:
        raise ValueError("window width must be positive")
    windows = []
    for boundary in boundaries:
        if boundary < 0 or boundary + width > len(changes):
            raise ValueError(f"incomplete boundary window at transition {boundary}")
        values = changes[boundary:boundary + width]
        peak = max(values)
        windows.append({"index": boundary, "immediate": float(values[0]),
                        "peak": float(peak), "peak_offset": values.index(peak),
                        "motion_sum": float(sum(values))})
    return windows


def summarize(windows: list[dict]) -> dict:
    """Summarize boundary windows without collapsing delayed peaks into seams."""
    if not windows:
        raise ValueError("no complete boundary windows")
    return {"count": len(windows),
            "mean_immediate": mean(w["immediate"] for w in windows),
            "mean_peak": mean(w["peak"] for w in windows),
            "max_peak": max(w["peak"] for w in windows),
            "mean_motion_sum": mean(w["motion_sum"] for w in windows),
            "delayed_peak_count": sum(w["peak_offset"] > 0 for w in windows)}


def third_summaries(windows: list[dict]) -> dict:
    """Keep absent horizon thirds explicit for short exploratory videos."""
    partitions = np.array_split(np.arange(len(windows)), 3)
    return {name: summarize([windows[int(i)] for i in indices]) if len(indices) else None
            for name, indices in zip(("early", "middle", "late"), partitions)}


def transition_changes(frames: np.ndarray) -> list[float]:
    """Mean absolute grayscale change for each adjacent frame pair."""
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("expected at least two grayscale frames")
    delta = np.abs(np.diff(frames.astype(np.int16), axis=0))
    return delta.mean(axis=(1, 2)).tolist()


def assert_compatible(reference: dict, candidate: dict) -> None:
    """Reject video comparisons with different source geometry or duration."""
    for key in ("width", "height", "frames", "fps"):
        if reference[key] != candidate[key]:
            raise ValueError(f"mismatched {key}: {reference[key]} != {candidate[key]}")


def video_metadata(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
               "-of", "json", str(path)]
    payload = json.loads(subprocess.check_output(command))
    stream = payload["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frames": int(stream["nb_frames"]), "fps": stream["r_frame_rate"]}


def decode_grayscale(path: Path, frames: int, width: int = 180,
                     height: int = 104) -> np.ndarray:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-vsync", "0", "-i",
               str(path), "-vf", f"scale={width}:{height}:flags=bicubic,format=gray",
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.check_output(command)
    pixels_per_frame = width * height
    if len(raw) != frames * pixels_per_frame:
        raise ValueError(f"decoded {len(raw) // pixels_per_frame} frames, expected {frames}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(frames, height, width)


def study_video(path: Path) -> dict:
    metadata = video_metadata(path)
    frames = decode_grayscale(path, metadata["frames"])
    changes = transition_changes(frames)
    windows = boundary_windows(changes, boundary_indices(metadata["frames"]))
    if not windows:
        raise ValueError("video does not contain a complete chunk boundary")
    thirds = third_summaries(windows)
    return {"source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": metadata, "grayscale_size": [180, 104],
            "changes": changes, "boundaries": windows,
            "summary": summarize(windows), "thirds": thirds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({label for label, _ in args.video}) != len(args.video):
        parser.error("video labels must be unique")
    results = {label: study_video(Path(path)) for label, path in args.video}
    reference = next(iter(results.values()))["metadata"]
    for result in results.values():
        assert_compatible(reference, result["metadata"])
    payload = {"contract": "grayscale MAE, 180x104, FFmpeg -vsync 0, "
                           "transition indices 20+32k, eight-transition windows",
               "videos": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({label: result["summary"] for label, result in results.items()},
                     indent=2))


if __name__ == "__main__":
    main()
