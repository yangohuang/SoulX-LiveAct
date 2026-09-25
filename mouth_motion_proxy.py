"""Measure coarse lower-face motion in a continuous SyncNet face crop."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MOUTH_BOX = (60, 90, 160, 150)  # x0, y0, x1, y1 in the 224x224 S3FD crop
UPPER_BOX = (60, 30, 160, 90)


def region_changes(frames: np.ndarray,
                   mouth_box: tuple[int, int, int, int] = MOUTH_BOX,
                   upper_box: tuple[int, int, int, int] = UPPER_BOX,
                   ) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("expected at least two grayscale video frames")

    def changes(box):
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= frames.shape[2] and 0 <= y0 < y1 <= frames.shape[1]):
            raise ValueError("region lies outside face crop")
        patch = frames[:, y0:y1, x0:x1].astype(np.int16)
        return np.abs(np.diff(patch, axis=0)).mean(axis=(1, 2))

    return changes(mouth_box), changes(upper_box)


def summarize_periods(mouth_changes: np.ndarray, upper_changes: np.ndarray,
                      fps: int = 25, segment_seconds: int = 30,
                      segment_count: int = 3) -> list[dict]:
    if len(mouth_changes) != len(upper_changes) or fps < 1 or segment_seconds < 1:
        raise ValueError("invalid region-change series or time layout")
    span = fps * segment_seconds
    if len(mouth_changes) + 1 < span * segment_count:
        raise ValueError("face crop is shorter than requested complete periods")
    results = []
    for index in range(segment_count):
        start, stop = index * span, (index + 1) * span - 1
        mouth = mouth_changes[start:stop]
        upper = upper_changes[start:stop]
        results.append({"start_second": index * segment_seconds,
                        "end_second": (index + 1) * segment_seconds,
                        "transitions": int(len(mouth)),
                        "mouth_mean": float(np.mean(mouth)),
                        "mouth_median": float(np.median(mouth)),
                        "upper_mean": float(np.mean(upper)),
                        "mouth_minus_upper_mean": float(np.mean(mouth - upper))})
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--segment-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    capture = cv2.VideoCapture(str(args.crop_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {args.crop_video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if abs(fps - 25) > 1e-3:
        raise ValueError(f"expected 25-FPS face crop, found {fps}")
    frames = []
    try:
        while True:
            okay, bgr = capture.read()
            if not okay:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    gray = np.stack(frames)
    mouth, upper = region_changes(gray)
    result = {"source": str(args.source_video),
              "source_sha256": sha256(args.source_video),
              "crop": str(args.crop_video), "crop_sha256": sha256(args.crop_video),
              "crop_frames": len(gray), "crop_fps": 25,
              "mouth_box_xyxy": MOUTH_BOX, "upper_box_xyxy": UPPER_BOX,
              "limitation": "fixed crop ROIs include expression, jaw/head motion and tracker error; not a phoneme-aligned lip score",
              "segments": summarize_periods(mouth, upper,
                                           segment_count=args.segment_count)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["segments"], indent=2))


if __name__ == "__main__":
    main()
