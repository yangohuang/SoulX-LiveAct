"""Recompute fixed-time SyncNet scores from its saved shift-distance matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score_distances(distances: np.ndarray, vshift: int = 15) -> dict:
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 2 * vshift + 1:
        raise ValueError("nonempty distance matrix must have 2*vshift+1 columns")
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix contains nonfinite values")
    means = matrix.mean(axis=0)
    index = int(np.argmin(means))
    return {"windows": int(len(matrix)), "best_offset_25fps_frames": vshift - index,
            "min_distance": float(means[index]),
            "confidence": float(np.median(means) - means[index])}


def score_segments(distances: np.ndarray, fps: int, segment_seconds: int,
                   segment_count: int, vshift: int = 15,
                   margin_frames: int = 0) -> list[dict]:
    if fps < 1 or segment_seconds < 1 or segment_count < 1:
        raise ValueError("invalid segment layout")
    span = fps * segment_seconds
    if margin_frames < 0 or 2 * margin_frames >= span:
        raise ValueError("invalid segment edge margin")
    if len(distances) > span * segment_count:
        raise ValueError("distance matrix extends beyond requested segments")
    results = []
    for index in range(segment_count):
        full_piece = distances[index * span:min((index + 1) * span, len(distances))]
        if len(full_piece) < 0.9 * span:
            raise ValueError(f"segment {index} has fewer than 90% of expected scored windows")
        piece = full_piece[margin_frames:len(full_piece) - margin_frames]
        if not len(piece):
            raise ValueError(f"segment {index} has no windows after edge margin")
        results.append({"start_second": index * segment_seconds,
                        "end_second": (index + 1) * segment_seconds,
                        "edge_margin_seconds": margin_frames / fps,
                        **score_distances(piece, vshift=vshift)})
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=Path, required=True,
                        help="NPZ with a `distances` array from SyncNet activesd.pckl")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--crop-video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--segment-count", type=int, required=True)
    parser.add_argument("--edge-margin-seconds", type=int, default=0,
                        help="Also score each segment after excluding this many seconds at both edges.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.distances) as data:
        distances = data["distances"]
    result = {
        "pipeline": "joonson/syncnet_python, S3FD face crop at 25 FPS; 31-shift distances",
        "source": str(args.source_video), "source_sha256": sha256(args.source_video),
        "crop": str(args.crop_video), "crop_sha256": sha256(args.crop_video),
        "model_sha256": sha256(args.model),
        "distance_matrix": str(args.distances),
        "distance_matrix_sha256": sha256(args.distances),
        "distance_matrix_shape": list(distances.shape),
        "full": score_distances(distances),
        "segments": score_segments(distances, fps=25, segment_seconds=30,
                                   segment_count=args.segment_count),
    }
    if args.edge_margin_seconds:
        result["interior_segments"] = score_segments(
            distances, fps=25, segment_seconds=30,
            segment_count=args.segment_count,
            margin_frames=25 * args.edge_margin_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"full": result["full"], "segments": result["segments"]}, indent=2))


if __name__ == "__main__":
    main()
