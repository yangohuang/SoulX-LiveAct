# Chunk-start jump diagnosis implementation plan

**Goal:** Attribute a 30-second LiveAct motion jump to denoiser output or
decoder behavior, then test a targeted low-cost mitigation on one RTX 4090.

**Architecture:** A default-off block capture in `generate.py` writes only
small latent slices and step metadata. A separate analysis script computes
temporal latent distances and relates them to the saved video frames. No
training or extra DiT forwards are introduced by the diagnostic run.

**Tech Stack:** PyTorch, existing LiveAct runtime, FFmpeg, unittest.

## Task 1: Capture helper and tests

- [x] Write a failing test for slicing the previous last two and current
  first two clean latents without changing either tensor.
- [x] Implement the helper in a focused module and rerun targeted tests.
- [x] Add a default-off `--boundary_probe_block` and output directory to
  `generate.py`, and store slices after each denoising step only for that
  block. Save final latent slices and metadata.

## Task 2: Identical-output validation

- [x] Run the existing unit suite and a short input with probe enabled.
- [x] Compare an instrumented short video to the corresponding baseline.
  Runs are not pixel-identical (32.76 dB PSNR); variation begins before the
  selected probe block. Avoid a bitwise-equality claim and interpret only
  same-run latent measurements.

## Task 3: Thirty-second diagnosis

- [x] Run the image-1 30-second baseline with capture at the block containing
  frame 373. Confirm 722 output frames and full decode.
- [x] Compute adjacent latent distances for the previous and new block at
  each step, and decoded-frame differences around frame 373→374.
- [x] Record which stage first exhibits an anomalous transition.

## Task 4: Targeted fix and evidence

- [x] Implement an optional spatial low-pass of the existing clean-latent
  anchor correction, selected because the probe found excessive low-frequency
  change in the generated latent and VAE context did not remove the jump.
- [x] Compare against no anchor and fixed anchor on the same inputs, including
  visible motion and lip-sync diagnostics.
- [x] Document outcomes, run `git diff --check` and tests, and sync only a
  personal-fork research branch. Do not open an upstream PR.

Result: the low-pass anchor moves the target hand jump from 373→374 to
376→377 and fails the second-identity gate. Keep it default-off as a negative
research result; do not describe it as a continuity fix.
