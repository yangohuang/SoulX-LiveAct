# Rollout-conditioned latent bridge implementation plan

**Goal:** Train and evaluate a small cross-block latent correction module on
one RTX 4090 without changing the 18B backbone or the PR-ready branch.

**Workspace:** `/home/yg/yg/code/github/SoulX/SoulX-LiveAct-rollout-bridge`.
**Runtime:** `/home/yg/miniforge3/envs/liveact/bin/python`.

## Task 1: Capture generated rollout latents

- [x] Write `tests/test_rollout_capture.py` for block indexing, CPU copy,
  non-mutation, and a round trip with `weights_only=True`; run it red.
- [x] Add a focused `rollout_capture.py` helper and `--rollout_latent_dir` to
  `generate.py`, saving final per-block latents only when enabled; run green.
- [x] Run capture-enabled rollouts for three identities, verifying block
  counts, 194-frame output, and successful video writes. A separate
  pixel-by-pixel paired run was omitted after the real-boundary quality gate
  failed; earlier boundary-probe paired runs showed cross-run variation even
  before probe execution.

## Task 2: Learnable bridge and synthetic-boundary dataset

- [x] Write failing tests for window extraction, corrupt/identity examples,
  output shape, zero-initialized no-op, non-mutating correction, and gradient
  flow to bridge parameters.
- [x] Implement the smallest 2D convolutional residual model and dataset in
  `rollout_bridge.py`; preserve the original six later latents on application.
- [x] Implement `train_rollout_bridge.py` with fixed split, seed, loss log,
  checkpoint metadata, and train/validation curves. No DiT or VAE gradients.

## Task 3: GPU data and pilot training

- [x] Capture baseline generated latents for identities 2 and 3 (training)
  and identity 4 (held-out validation), with the prior 4090 runtime flags.
- [x] Train at most a small bridge on the RTX 4090; record wall time, peak
  VRAM, train and held-out losses, and checkpoint hash.
- [x] Reject if the held-out synthetic corruption loss does not beat the
  zero-correction baseline or if uncorrupted validation samples drift. The
  synthetic score improved, but clean-input drift exceeded the absolute gain.

## Task 4: True-boundary evaluation and decision

- [x] Decode image-1 block 12 with the trained bridge, using the captured
  prior five and current eight final latents. Compare all transitions from
  frames 371–381, not only 373→374, with baseline and anchors.
- [x] Apply the bridge to three held-out identity-4 boundaries and compare
  visual motion. There is no short integrated output to score with SyncNet.
- [x] The offline gates failed, so stop before adding an inference flag or
  running full-video/SyncNet checks. Document the negative result.
- [x] Run the full unit suite, `git diff --check`, and save a reproducible
  report. Sync only the personal-fork research branch.
