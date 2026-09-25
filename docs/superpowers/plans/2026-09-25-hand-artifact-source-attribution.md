# Hand Artifact Source Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requested a main agent to coordinate these experiments, so implementation stays inline in the existing isolated worktree.

**Goal:** Separate the final velocity-anchor correction from VAE history sensitivity at a reproducible hand-artifact block.

**Architecture:** Reuse the existing block probe and VAE decoder; add one opt-in full pre-anchor final snapshot. A small replay evaluator validates self-arm reconstruction, compares pre/post correction with fixed history, then runs the secondary 2×2 history/current swap.

**Tech Stack:** PyTorch, LightX2V WanVAE, FFmpeg, RTX 4090, pytest.

---

### Task 1: Snapshot and replay validation

**Files:** `generate.py`, `tests/test_boundary_probe.py`, `redecode_boundary.py`

- [x] Reuse the existing `capture_final_context` test, which already checks that prior-five and all current latents are retained for VAE replay.
- [x] Save `block-16-pre-anchor-final.pt` only for the selected block at the final step before `anchor_chunk_start`; do not alter default generation.
- [x] Implement and test a pure helper that assembles prior-last-three plus current-eight and checks matching geometry.

### Task 2: 4090 captures and VAE replay

**Files:** `docs/benchmarks/artifacts/*`, `docs/benchmarks/2026-09-25-hand-artifact-source-attribution.md`

- [x] Run seed-43 identity-2 unanchored and velocity arms sequentially with block-16 probes; verify 722-frame videos, snapshot shapes and hashes.
- [x] Decode the two self-arm pairs, compare frames 501–508 with their generated MP4s; if acceptable, decode velocity pre/post final-step correction with fixed prior history.
- [x] Decode the four history/current combinations, save visible rows and RGB seam/first-eight changes; state the cross-arm trajectory caveat.

### Task 3: Evidence and integration

**Files:** study report, roadmap, interview evidence, memory

- [x] Record source/input hashes, tests, visual conclusion, decode latency and limitations; do not claim a general repair.
- [ ] Run full tests and video/JSON verification, then commit and sync only the personal fork; verify remote SHA and clean worktree.
