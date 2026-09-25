"""Sample face-recognition consistency; never store biometric embeddings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("embedding dimensions differ")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("zero face embedding")
    return float(np.dot(left, right) / denominator)


def summarize_embeddings(embeddings: list[np.ndarray | None],
                         reference: np.ndarray) -> dict:
    if not embeddings:
        raise ValueError("no sampled video frames")
    first = next((value for value in embeddings if value is not None), None)
    if first is None:
        raise ValueError("no faces detected in sampled video")
    reference_cosine = [cosine(value, reference) if value is not None else None
                        for value in embeddings]
    first_cosine = [cosine(value, first) if value is not None else None
                    for value in embeddings]
    found = [value for value in reference_cosine if value is not None]
    split = max(1, len(embeddings) // 3)
    early = [value for value in reference_cosine[:split] if value is not None]
    late = [value for value in reference_cosine[-split:] if value is not None]
    coverage = len(found) / len(embeddings)
    early_coverage = len(early) / split
    late_coverage = len(late) / split
    return {"samples": len(embeddings), "detected": len(found),
            "coverage": coverage, "early_coverage": early_coverage,
            "late_coverage": late_coverage,
            "score_usable": min(coverage, early_coverage, late_coverage) >= 0.9,
            "first_detected_sample_index": next(i for i, value in enumerate(embeddings)
                                                if value is not None),
            "reference_cosine": reference_cosine,
            "first_cosine": first_cosine,
            "mean_reference_cosine": float(np.mean(found)),
            "p10_reference_cosine": float(np.percentile(found, 10)),
            "early_mean_reference_cosine": float(np.mean(early)) if early else None,
            "late_mean_reference_cosine": float(np.mean(late)) if late else None}


def detect_single_face_embedding(analyzer, bgr_frame: np.ndarray):
    faces = analyzer.get(bgr_frame)
    if len(faces) != 1:
        return None, None, len(faces)
    face = faces[0]
    return np.asarray(face.embedding, dtype=np.float32), float(face.det_score), 1


def sample_video(analyzer, path: Path, frame_stride: int = 24) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    embeddings, scores, counts, frame_indices = [], [], [], []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                embedding, score, count = detect_single_face_embedding(analyzer, frame)
                embeddings.append(embedding)
                scores.append(score)
                counts.append(count)
                frame_indices.append(frame_index)
            frame_index += 1
    finally:
        capture.release()
    return {"embeddings": embeddings, "detection_scores": scores,
            "detected_face_counts": counts,
            "frame_indices": frame_indices, "decoded_frames": frame_index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path,
                        default=Path("/home/yg/.insightface"))
    args = parser.parse_args()

    import cv2
    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(name="buffalo_l", root=str(args.model_root),
                            allowed_modules=["detection", "recognition"],
                            providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    reference_image = cv2.imread(str(args.reference))
    if reference_image is None:
        raise ValueError(f"cannot read reference: {args.reference}")
    reference, reference_score, reference_count = detect_single_face_embedding(
        analyzer, reference_image)
    if reference is None:
        raise ValueError(f"reference must have exactly one face, found {reference_count}")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    model_dir = args.model_root / "models" / "buffalo_l"
    model_hashes = {name: sha256(model_dir / name)
                    for name in ("det_10g.onnx", "w600k_r50.onnx")}

    output = {"reference": str(args.reference),
              "reference_sha256": sha256(args.reference),
              "reference_detection_score": reference_score,
              "model": "InsightFace buffalo_l/w600k_r50, CPU, 640x640 detector",
              "model_sha256": model_hashes,
              "sample_stride_frames": 24, "videos": {}}
    for label, path in args.video:
        video_path = Path(path)
        sample = sample_video(analyzer, video_path)
        scores = summarize_embeddings(sample.pop("embeddings"), reference)
        output["videos"][label] = {"source": str(path),
                                   "sha256": sha256(video_path), **sample, **scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({label: {key: value for key, value in item.items()
                              if key in ("coverage", "mean_reference_cosine",
                                         "p10_reference_cosine",
                                         "early_mean_reference_cosine",
                                         "late_mean_reference_cosine")}
                      for label, item in output["videos"].items()}, indent=2))


if __name__ == "__main__":
    main()
