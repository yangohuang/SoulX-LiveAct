# Pre-output Boundary Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requested that one main agent run the experiments, so execution remains inline in the current isolated worktree.

**Goal:** Record cheap pre-output clean-latent continuity scalars and decide from two 4090 rollouts whether they are useful artifact triggers.

**Architecture:** Add a pure tensor metric helper in `boundary_probe.py` and an opt-in JSON recorder in `generate.py`; leave model and sampling behavior unchanged. A separate analysis script scores the generated videos and joins the boundary series for a prespecified negative/positive decision.

**Tech Stack:** Python, PyTorch, pytest/unittest, FFmpeg, OpenCV, RTX 4090.

---

### Task 1: Pure metric

**Files:** `boundary_probe.py`, `tests/test_boundary_probe.py`

- [x] Write a test with previous scalar latents `[1,3]` and predicted first latent `5`: gap 2, previous speed 2, velocity residual 0; also check a mismatched shape raises `ValueError`.
- [x] Run `PYTHONPATH=/tmp/liveact-pytest-overlay /home/yg/miniforge3/envs/liveact/bin/python -m pytest -q tests/test_boundary_probe.py` and observe the missing-symbol failure.
- [x] Implement `pre_output_boundary_metrics(previous, predicted)` returning three 0D tensors from float32 differences without a CPU transfer.
- [x] Rerun the focused test; inspect gradient-free behavior and syntax.

### Task 2: Opt-in recorder

**Files:** `generate.py`, `tests/test_boundary_probe.py`

- [x] Write a serialization test for one synthetic block's first/final step, frame mapping and finite JSON values; observe the missing-recorder failure.
- [x] Add `--boundary_signal_dir`; collect detached scalars at first/final denoising steps before anchoring, then write one JSON per request after the video is exported.
- [x] Rerun focused and full test suites; verify default generation path has no recording operations.

### Task 3: Two-rollout study

**Files:** `docs/benchmarks/2026-09-25-pre-output-boundary-signal-study.md`, `docs/benchmarks/artifacts/*`

- [x] Run identity 2 seed 42 and 43 unanchored and seed 43 velocity-anchor for 30 seconds on one 4090, recording video, log and JSON for each.
- [x] Verify videos decode to 722 frames, compare SHA-256 with prior baseline, and score all 22 boundaries using the same first-eight-transition grayscale MAE method.
- [x] Calculate rank association and top-five overlap, review selected visual windows, apply the prespecified gate, and document result/cost limitations.

### Task 4: Integration

**Files:** roadmap, interview evidence card, workspace memory

- [x] Run full tests and artifact verification once more; review git diff and ensure no official PR.
- [x] Update roadmap/interview record with the result, push this isolated branch to the personal fork, verify remote SHA and clean state.
