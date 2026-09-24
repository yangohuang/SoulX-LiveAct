# Motion-trend boundary conditioning implementation plan

**Goal:** Test whether propagating recent latent motion improves LiveAct chunk
continuity on one RTX 4090 without additional denoising passes.

**Architecture:** Extend the existing opt-in `anchor_chunk_start` helper with a
trend coefficient and expose it in `generate.py`. Keep defaults identical, and
record comparative media, objective proxies, and qualitative limitations.

**Tech Stack:** Python, PyTorch, unittest, FFmpeg, existing LiveAct FP8 runtime.

## Task 1: Trend-target helper

- [x] Add unit tests in `tests/test_temporal_continuity.py` for zero-trend
  compatibility, first- and second-frame extrapolation, invalid trend values,
  insufficient history, and unchanged later frames. Run the tests and observe
  the expected failure before editing production code.
- [x] Extend `temporal_continuity.py:anchor_chunk_start` with `trend=0.0` and
  `target_k = previous[-1] + k * trend * (previous[-1]-previous[-2])` for
  `k=1,2`, preserving the existing blend weights.
- [x] Run the targeted unit tests, then the relevant existing unit suite.

## Task 2: Runtime option

- [x] Add `--motion_anchor_trend` to `generate.py`, default 0, allowed range
  `[0,1]`, and reject trend without positive anchor strength.
- [x] Pass the trend coefficient to `anchor_chunk_start` at the existing
  predicted-clean-latent hook. Add a focused argument-validation test if the
  runtime parser exposes one; otherwise verify with `generate.py --help` and
  a parser-only invocation.
- [x] Document the option and its experimental quality caveat in `README.md`.

## Task 3: Controlled single-card experiment

- [x] Verify the RTX 4090 is idle enough and reuse the cached FP8/prompt inputs.
- [x] Run identical short inputs for both example identities with no anchor,
  fixed anchor at strength 0.25, and trend anchor at strength 0.25 with trend
  0.5. Keep 3 denoising steps and output resolution 416×720.
- [x] Decode each MP4 fully, compute first-eight-frame and later-frame motion
  proxies at each 32-frame boundary, and produce side-by-side review videos.
- [ ] Complete human audiovisual review of motion naturalness and mouth
  articulation. This requires viewing and listening to the comparison clips;
  automated proxies and SyncNet scores do not substitute for it.
- [x] Apply the 30-second gate: the trend pilot did not beat the fixed anchor
  on both identities, so do not spend GPU time on a longer confirmation run.

## Task 4: Evidence and fork sync

- [x] Record exact commands, GPU, timings, metrics, media paths, and limitations
  in a benchmark note. State clearly that this is inference-time conditioning,
  not Vidu Self-Replay Forcing or Vivix's unpublished implementation.
- [ ] Run `git diff --check`, targeted tests, and MP4 decode checks. Commit the
  isolated research branch and push only to the user's personal fork. Do not
  open an upstream PR.
