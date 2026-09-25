"""Small, default-off fingerprints for pre-encoder LiveAct video frames."""

import hashlib

import numpy as np


def exporter_frame_sha256(frames: np.ndarray) -> list[str]:
    """Hash exactly the RGB bytes Diffusers passes to imageio's video writer."""
    video = np.asarray(frames)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("expected T×H×W×3 RGB frames")
    quantized = (video * 255).astype(np.uint8)
    return [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
            for frame in quantized]


def first_divergence(left: list[str], right: list[str]) -> int | None:
    """Return the first different frame in the shared prefix, if any."""
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None)
