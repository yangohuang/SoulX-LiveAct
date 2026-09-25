"""Record in-frame upper-body pose landmarks for descriptive motion audits."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from long_horizon_stability import assert_compatible, video_metadata


LANDMARKS = {"left_shoulder": 11, "right_shoulder": 12,
             "left_elbow": 13, "right_elbow": 14,
             "left_wrist": 15, "right_wrist": 16}


def is_valid(point: dict | None, min_visibility: float = 0.5) -> bool:
    return (point is not None and point["visibility"] >= min_visibility
            and 0 <= point["x"] <= 1 and 0 <= point["y"] <= 1)


def displacements(landmarks: list[dict | None],
                  min_visibility: float = 0.5) -> list[float | None]:
    """Return normalized adjacent-frame travel only for two visible in-frame points."""
    changes = []
    for before, after in zip(landmarks[:-1], landmarks[1:]):
        if is_valid(before, min_visibility) and is_valid(after, min_visibility):
            changes.append(math.hypot(after["x"] - before["x"],
                                      after["y"] - before["y"]))
        else:
            changes.append(None)
    return changes


def analyze_video(path: Path) -> dict:
    import cv2
    import mediapipe as mp

    metadata = video_metadata(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    trajectories = {name: [] for name in LANDMARKS}
    try:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    smooth_landmarks=True,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5) as detector:
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = result.pose_landmarks.landmark if result.pose_landmarks else None
                for name, index in LANDMARKS.items():
                    point = points[index] if points else None
                    trajectories[name].append(
                        {"x": float(point.x), "y": float(point.y),
                         "visibility": float(point.visibility)} if point else None)
    finally:
        capture.release()
    if any(len(points) != metadata["frames"] for points in trajectories.values()):
        raise ValueError(f"decoded frame count differs from metadata for {path}")
    summaries = {}
    for name, points in trajectories.items():
        changes = displacements(points)
        valid = [change for change in changes if change is not None]
        summaries[name] = {"valid_frames": sum(is_valid(point) for point in points),
                           "valid_transitions": len(valid),
                           "mean_valid_displacement": sum(valid) / len(valid) if valid else None}
    return {"source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": metadata, "trajectories": trajectories, "summary": summaries}


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
    payload = {"method": "MediaPipe Pose 0.10.11, model_complexity=1, tracker; "
                         "visibility>=0.5 and in-frame points only; normalized image coordinates",
               "videos": videos}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: video["summary"] for name, video in videos.items()}, indent=2))


if __name__ == "__main__":
    main()
