# Exact Audio Conditioning Horizon Implementation Plan

> **For agentic workers:** Execute inline in this isolated research worktree. The user authorized local 4090 experiments and personal-fork sync; no official PR.

**Goal:** Repeat the exact model-received 30-second audio features across a 90-second causal LiveAct rollout and test for late quality loss against a fresh 30-second control.

**Architecture:** Add a default-off feature-tiling helper and CLI switch, with SHA-256 sidecars proving period/context equality. Use the existing three-step generation and horizon evaluation tools without changing model weights or the generated-history path.

**Tech Stack:** Python, PyTorch BF16, RTX 4090, SoX/Wav2Vec, FFmpeg, SyncNet, InsightFace CPU, pytest.

---

- [x] Write failing tests for exact feature tiling, invalid target lengths and equal interior ±2-frame-context fingerprints across three periods. Run focused tests red.
- [x] Implement the pure tiling/fingerprint helper; run focused tests green.
- [x] Add `--audio_embedding_repeat_source` to `generate.py`, reject with streaming-audio mode, encode source WAV through the same resampler/Wav2Vec, tile to the normal target feature length, write a sidecar JSON by the output video and pass the new flag through the monitored harness. Keep no-flag behavior unchanged.
- [x] Run syntax/full regression tests and a 5-second real CLI smoke with source=input. Verify full decode, sidecar hash/shape, peak GPU memory and a sparse contact sheet.
- [x] Freeze archived image/source/90-second WAV hashes, seed43 runtime options and threshold. Run the 90-second exact-feature rollout on the 4090; monitor 68 blocks and safety memory samples.
- [x] Run a separate 30-second fresh-history control under the same flag. Verify its canonical feature hash equals each long-run period's hash; compare full and interior ±2-context hashes.
- [x] Compute 90-second segmented SyncNet (full and 2-second-edge-trimmed), face, boundary motion and mouth-motion proxies, plus matched-time visual strips/worst windows. Apply the written gate; only plan seed44 if all checks pass.
- [x] Archive videos, sidecars, score matrices, charts, raw JSON and SHA-256 manifest. Write a concise report and update the Vivix evidence card/`MEMORY.md` without overstating causal history or real-time performance.
- [x] Verify tests, hashes, report links and clean diff; commit and sync only the personal fork, then verify remote hash and clean local branch.
