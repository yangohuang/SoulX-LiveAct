# Long-horizon stability study implementation plan

> **For agentic workers:** Execute tasks inline in this session. Each checklist item has a concrete verification gate.

**Goal:** Produce a reproducible 30-second stability diagnosis that detects shifted or suppressed motion rather than rewarding a smaller single seam value.

**Architecture:** A CPU-only evaluator decodes three MP4 files to matched grayscale frames, computes transition traces and per-boundary windows, and writes JSON. A report interprets these numbers alongside existing SyncNet and visual evidence. No model code is changed.

**Tech stack:** Python standard library, NumPy, FFmpeg/FFprobe, unittest.

---

### Task 1: Metric contract

**Files:** `tests/test_long_horizon_stability.py`, `long_horizon_stability.py`

- [x] Write a failing test with a synthetic transition trace containing a delayed spike; require the evaluator to return its offset, window peak, and total motion.
- [x] Run `/home/yg/miniforge3/envs/digithuman/bin/python -m unittest tests.test_long_horizon_stability -v` and verify the missing API causes failure.
- [x] Implement `boundary_windows(changes, boundaries, width=8)` and `summarize(windows)` with explicit index validation and no smoothing.
- [x] Re-run the focused test and add mismatch/empty-input cases before refactoring.

### Task 2: Video adapter and fixed experiment

**Files:** `long_horizon_stability.py`, `tests/test_long_horizon_stability.py`, `docs/benchmarks/artifacts/2026-09-25-long-horizon-stability.json`

- [x] Write a failing test for deriving boundaries from 722 frames: 22 indices, first 20, last 692.
- [x] Implement FFprobe metadata checking, FFmpeg `-vsync 0` grayscale decode at 180×104, transition MAE, and the CLI for three labeled inputs.
- [x] Run on the existing baseline, anchor 0.25 and anchor 0.45 MP4 files; save JSON with source hashes and per-transition/per-boundary traces.
- [x] Check source frame counts and reconcile metric differences with the earlier study: these three files are separately generated and the new report explicitly uses the saved JSON and source hashes, without treating older summary values as exact expected outputs.

### Task 3: Research interpretation

**Files:** `docs/benchmarks/2026-09-25-long-horizon-stability-study.md`, `docs/benchmarks/artifacts/2026-09-25-long-horizon-stability.json`

- [x] Write the early/middle/late comparison, delayed-spike check, lip-sync tradeoff, and limitations.
- [x] Link the existing visual samples and document the replication gate: three identities, two seeds, same audio/prompt settings, and explicit identity metric before any improvement claim.
- [x] Run focused and full tests (48 passing with GPU access), JSON content checks and `git diff --check`; commit and sync only the personal fork.
