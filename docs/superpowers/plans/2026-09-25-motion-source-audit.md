# Spatial Motion-Source Audit Implementation Plan

> **For agentic workers:** Execute inline in the existing research worktree. The user authorized autonomous local analysis and personal-fork sync; no official PR.

**Goal:** Localize the seed-44 motion-retention failure of the first-audio two-step policy without changing the original quality decision.

**Architecture:** Reuse the existing FFmpeg grayscale decoder and fixed chunk indices. One new CPU-only script measures adjacent-frame grayscale difference and Farnebäck flow in four frozen spatial masks. Compare the six archived videos at identical frame count and resolution and preserve complete timelines plus summaries.

**Tech Stack:** Python, NumPy, OpenCV, FFmpeg, pytest and matplotlib.

---

- [x] Write `tests/test_spatial_motion_audit.py` with a textured square moving only in the lower band; assert lower-band grayscale and flow respond while upper/mouth/background stay near zero. Add a 722-frame assertion for 22 exact boundary indices from 20 to 692.
- [x] Run the focused test with `PYTHONPATH=/tmp/liveact-pytest-overlay:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yg/miniforge3/envs/liveact/bin/python -m pytest -q tests/test_spatial_motion_audit.py` and confirm it fails because the module is missing.
- [x] Implement `spatial_motion_audit.py` with fixed 104×180 masks, `measure_pair()`, six-video CLI, per-transition and boundary-window JSON; validate matching metadata and save SHA-256 source hashes. Add an exploratory visibility-filtered Pose script after the spatial result suggested time redistribution.
- [x] Run focused and full tests (84 passed); create a region overlay on a decoded source frame and inspect it before interpreting metrics.
- [x] Run six-video analysis, compute paired ROI ratios/paired-window counts and plot by seed and arm; compare whole-frame downscaled MAE direction against the original 180×104 study.
- [x] Write `docs/benchmarks/2026-09-25-spatial-motion-source-audit.md`, archive JSON/figure/overlay and hash manifest, update Vivix evidence and `MEMORY.md` while preserving the original gate failure.
- [x] Verify links, hashes, tests and clean diff; commit and sync only the personal fork, then verify the remote hash and clean branch.
