"""Reproducible pixel-change proxy for LiveAct chunk boundaries.

This is a diagnostic for abrupt visual changes, not a measure of correct
motion, identity, or audio alignment.
"""

import argparse
import json
import subprocess

import numpy as np


def boundary_metrics(frames: np.ndarray, first_boundary: int = 20,
                     block_frames: int = 32, opening_window: int = 8) -> dict:
    """Compare adjacent-frame changes just after each chunk boundary."""
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("frames must have shape (time, height, width) and contain two frames")
    if first_boundary < 0 or block_frames < 1 or not 0 < opening_window <= block_frames:
        raise ValueError("invalid boundary layout")

    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(axis=(1, 2))
    seams = []
    first_windows = []
    later_windows = []
    per_boundary = []
    for index in range(first_boundary, len(diffs), block_frames):
        first = diffs[index:index + opening_window]
        if len(first) < opening_window:
            break
        later = diffs[index + opening_window:index + block_frames]
        seams.append(float(diffs[index]))
        first_windows.append(float(first.mean()))
        if len(later) == block_frames - opening_window:
            later_windows.append(float(later.mean()))
        per_boundary.append({"transition_index": index, "first_window_mean": float(first.mean()),
                             "seam": float(diffs[index])})

    return {"frame_count": len(frames), "boundary_count": len(seams),
            "first_window_mean": float(np.mean(first_windows)) if first_windows else None,
            "later_window_mean": float(np.mean(later_windows)) if later_windows else None,
            "seam_mean": float(np.mean(seams)) if seams else None,
            "worst_seam": float(max(seams)) if seams else None,
            "per_boundary": per_boundary}


def decode_grayscale(path: str, width: int = 180, height: int = 104) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf",
                             f"scale={width}:{height},format=gray", "-f", "rawvideo", "-pix_fmt",
                             "gray", "-"], capture_output=True, check=True)
    frame_bytes = width * height
    if len(result.stdout) % frame_bytes:
        raise ValueError("decoded byte count is not a whole number of frames")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    args = parser.parse_args()
    print(json.dumps(boundary_metrics(decode_grayscale(args.video)), indent=2))
