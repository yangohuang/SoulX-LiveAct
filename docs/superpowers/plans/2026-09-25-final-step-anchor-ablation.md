# Final-Step Velocity Anchor Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requested one main agent to coordinate all experiments, so execution remains inline in the isolated worktree.

**Goal:** Test whether skipping only the final velocity-anchor correction avoids a proven hand artifact in a full causal 30-second rollout without sacrificing motion or lip quality.

**Architecture:** Add a pure per-step gate in `temporal_continuity.py` and one default-off CLI flag in `generate.py`; retain the existing default all-step path. Reuse the established video/motion/SyncNet/face evaluators and visual gate before held-out work.

**Tech Stack:** Python, PyTorch, FFmpeg, local RTX 4090, SyncNet, InsightFace.

---

### Task 1: Step gate

**Files:** `temporal_continuity.py`, `generate.py`, `tests/test_temporal_continuity.py`

- [x] Add a failing test showing strength 0 disables every step, the existing mode enables every step, and opt-in skip-last enables steps 0/1 but disables step 2 under three-step sampling.
- [x] Implement the pure gate with index validation; expose `--motion_anchor_skip_last_step` and use it only at the existing anchor call site.
- [x] Run focused/full tests and syntax checks; verify no other generation arguments change.

### Task 2: Development video

**Files:** `docs/benchmarks/artifacts/*`, `docs/benchmarks/2026-09-25-final-step-anchor-ablation.md`

- [x] Generate image 2/seed 43 for 30 seconds on one 4090; verify decoded frames, hash, and baseline/velocity settings.
- [x] Score the 22 boundary windows and review known hand-failure frames plus top-five windows. Stop if visual gate fails.
- [x] If visual gate passes, run matched-crop SyncNet and face-reference evaluator, apply motion/lip gates; only then run held-out identity 3. The motion gate failed, so no held-out run was made.

### Task 3: Evidence and fork

**Files:** report, roadmap, interview evidence, memory

- [x] Save the playable video, JSON, contact sheets, commands and source hashes; clearly state a pass or rejection.
- [x] Run final verification, commit and push only to personal fork; verify remote SHA and clean worktree. No official PR.
