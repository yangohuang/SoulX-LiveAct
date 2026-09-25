"""Exact-repeat controls and fingerprints for LiveAct audio feature experiments."""

import hashlib

import torch


def repeat_embedding(source: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Tile one nonempty feature period to an integer-multiple target length."""
    if source.ndim < 2 or source.shape[0] < 1:
        raise ValueError("source must be a nonempty time-major feature tensor")
    if target_frames < 1 or target_frames % source.shape[0]:
        raise ValueError("target frames must be a positive multiple of source frames")
    repeats = target_frames // source.shape[0]
    return source.repeat((repeats,) + (1,) * (source.ndim - 1))


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact raw tensor bytes without converting BF16 values."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def period_fingerprints(embedding: torch.Tensor, period_frames: int,
                        margin_frames: int = 48,
                        context_radius: int = 2) -> dict:
    """Fingerprint each period and the feature span feeding interior ±radius windows."""
    if embedding.ndim < 2 or period_frames < 1 or len(embedding) % period_frames:
        raise ValueError("embedding length must be a positive multiple of period frames")
    if (context_radius < 0 or margin_frames <= context_radius or
            2 * margin_frames >= period_frames):
        raise ValueError("margin must exceed context radius and leave interior frames")
    raw_hashes, context_hashes = [], []
    for start in range(0, len(embedding), period_frames):
        raw_hashes.append(tensor_sha256(embedding[start:start + period_frames]))
        context_hashes.append(tensor_sha256(
            embedding[start + margin_frames - context_radius:
                      start + period_frames - margin_frames + context_radius]))
    return {"period_frames": period_frames, "repeat_count": len(raw_hashes),
            "interior_margin_frames": margin_frames,
            "context_radius_frames": context_radius,
            "period_sha256": raw_hashes,
            "interior_context_sha256": context_hashes}
