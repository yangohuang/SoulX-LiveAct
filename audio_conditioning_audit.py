"""Audit whether repeated raw speech yields matched LiveAct Wav2Vec conditioning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare_periods(left: np.ndarray, right: np.ndarray,
                    margin_frames: int = 48) -> dict:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("conditioning periods must have the same frame and feature shape")
    if margin_frames < 0 or 2 * margin_frames >= len(left):
        raise ValueError("invalid edge margin")
    stop = len(left) - margin_frames
    first = np.asarray(left[margin_frames:stop], dtype=np.float32).reshape(-1, np.prod(left.shape[1:]))
    second = np.asarray(right[margin_frames:stop], dtype=np.float32).reshape(first.shape)
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    if np.any(denominator == 0):
        raise ValueError("conditioning contains a zero feature vector")
    cosine = numerator / denominator
    return {"frames": len(first), "mean_cosine": float(np.mean(cosine)),
            "p10_cosine": float(np.percentile(cosine, 10)),
            "min_cosine": float(np.min(cosine)),
            "normalized_mae": float(np.mean(np.abs(first - second)) /
                                    np.mean(np.abs(first)))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-audio", type=Path, required=True)
    parser.add_argument("--fresh-audio", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torchaudio
    import torchaudio.transforms as T
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model
    from util_liveact import get_embedding

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Wav2Vec2Model.from_pretrained(
        str(args.model_dir), local_files_only=True, torch_dtype=dtype
    ).to(args.device, dtype=dtype).eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(args.model_dir), local_files_only=True)
    model.feature_extractor._freeze_parameters()

    def encode(path: Path) -> np.ndarray:
        raw, source_rate = torchaudio.load(str(path))
        stretched, rate = torchaudio.sox_effects.apply_effects_tensor(
            raw, source_rate, [["tempo", f"{25 / 24}"]])
        samples = T.Resample(rate, 16000)(stretched) * 3.0
        with torch.no_grad():
            embedding = get_embedding(samples[0], feature_extractor, model,
                                      device=args.device)
        return embedding.to("cpu", dtype=torch.float32).numpy()

    repeated = encode(args.repeat_audio)
    fresh = encode(args.fresh_audio)
    if len(repeated) != 3 * len(fresh):
        raise ValueError(f"expected 3x embedding frames, got {len(repeated)} and {len(fresh)}")
    period = len(fresh)
    pieces = [repeated[i * period:(i + 1) * period] for i in range(3)]
    result = {"repeat_audio": str(args.repeat_audio),
              "repeat_audio_sha256": sha256(args.repeat_audio),
              "fresh_audio": str(args.fresh_audio),
              "fresh_audio_sha256": sha256(args.fresh_audio),
              "wav2vec_weights_sha256": sha256(args.model_dir / "pytorch_model.bin"),
              "preprocessing": "generate.py resample_audio: SoX tempo=25/24, 16kHz resample, x3 amplitude; get_embedding, BF16 CUDA",
              "device": args.device, "dtype": str(dtype),
              "embedding_shape_repeat": list(repeated.shape),
              "embedding_shape_fresh": list(fresh.shape),
              "margin_frames": 48,
              "repeat_1_vs_0": compare_periods(pieces[0], pieces[1]),
              "repeat_2_vs_0": compare_periods(pieces[0], pieces[2]),
              "fresh_vs_repeat_0": compare_periods(fresh, pieces[0])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in
                      ("embedding_shape_repeat", "embedding_shape_fresh",
                       "repeat_1_vs_0", "repeat_2_vs_0", "fresh_vs_repeat_0")}, indent=2))


if __name__ == "__main__":
    main()
