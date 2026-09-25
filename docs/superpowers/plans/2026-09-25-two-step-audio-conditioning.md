# Two-Step First-Audio Conditioning Implementation Plan

> **For agentic workers:** Execute the checked steps inline in this already isolated research worktree. The user authorized autonomous experimentation and personal-fork sync; SoulX official PR remains out of scope.

**Goal:** Test whether audio conditioning on both calls of LiveAct's two-step schedule recovers lip synchronization while retaining its one-card speed gain.

**Architecture:** Extend `denoising_schedule` with one default-off policy argument and pass an opt-in CLI flag from `generate.py`; use the existing monitored harness to launch the controlled video. Keep all model code and default schedules unchanged. Evaluate the new arm against the already archived deterministic controls.

**Tech Stack:** Python 3.10, PyTorch/CUDA on RTX 4090, pytest, FFmpeg, SyncNet, InsightFace CPU, existing LiveAct evaluation scripts.

---

### Task 1: Lock the schedule behavior with tests

**Files:** `tests/test_denoising_schedule.py`, `denoising_schedule.py`

- [x] Add `test_two_steps_audio_first_step_conditions_both_calls` asserting `denoising_schedule(2, audio_first_step=True) == ((1000.0, 833.33333333, 0.0), (False, False))`.
- [x] Add `test_audio_first_step_rejects_three_steps` asserting a `ValueError` when `denoising_schedule(3, audio_first_step=True)`.
- [x] Run `PYTHONPATH=/tmp/liveact-pytest-overlay:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yg/miniforge3/envs/liveact/bin/python -m pytest -q tests/test_denoising_schedule.py` and verify the two new tests fail for the missing argument.
- [x] Implement the default-off schedule parameter and rejection; re-run the focused tests and the existing default-schedule assertions.

### Task 2: Expose the experiment only through an explicit flag

**Files:** `generate.py`, `benchmark_pin_memory.py`, `tests/test_denoising_schedule.py`

- [x] Add `--audio_first_step` (`store_true`, default false) to the generation CLI and pass it to `denoising_schedule(args.denoising_steps, audio_first_step=args.audio_first_step)`; the schedule validation must happen before model loading.
- [x] Add `--audio-first-step` to the monitored harness and append `--audio_first_step` only when set. Keep the existing command record in resources JSON.
- [x] Run `python -m py_compile denoising_schedule.py generate.py benchmark_pin_memory.py`, `git diff --check`, and the full repository test suite with GPU-visible vLLM device discovery (80 passed).

### Task 3: Run the preregistered GPU gates

**Files:** `docs/benchmarks/2026-09-25-two-step-audio-conditioning.md`, `docs/benchmarks/artifacts/2026-09-25-two-step-audio-conditioning/`

- [x] Confirm exact seed-43 control hashes, input PCM hash and 4090 headroom from the archived deterministic recheck.
- [x] Run a 5-second 416×720 smoke test using image 2/seed 43 and `--disable_cudnn_benchmark --audio_first_step`; it yielded 117 frames and a 4.864-second output audio track from the 5-second PCM input, with no obvious gross defect. This source-to-output trim is documented rather than silently called a five-second output.
- [x] If smoke passes, run the exact 30-second arm, monitor all 23 block times, and confirm 722 decoded frames with 30-second audio.
- [x] Compute motion, once-per-second face proxy, 747×31 SyncNet distance matrix, contact sheet and worst-five boundary sheets. Seed 43 passes every threshold without tuning.
- [x] Run seed 44 only after seed 43 passed all development gates. Seed 44 fails the motion-retention gate at 78.2%; stop without an identity holdout or threshold tuning.

### Task 4: Review and sync

**Files:** report, roadmap, interview evidence card, `MEMORY.md`

- [x] Archive video/audio/JSON/figures and SHA-256 manifest; link playable video in the report, state the exact decision and limitations.
- [x] Update the research roadmap and Vivix interview evidence with the real result; leave resume v50 unchanged.
- [x] Verify all report links, artifact hashes, decode metadata, test suite and clean diff; commit and push only `fork/codex/liveact-long-horizon-study`, then verify remote hash and clean branch.
