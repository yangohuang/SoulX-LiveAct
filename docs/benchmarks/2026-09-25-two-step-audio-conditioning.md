# Two-step first-audio conditioning on one RTX 4090

The [deterministic step frontier](2026-09-25-deterministic-step-quality-recheck.md) rejected the stock two-step schedule: seed 43 lost 0.667 SyncNet confidence versus three steps despite a 30.3% later-block speed gain, and seed 44 repeated a 0.430 loss. The stock schedules change both timesteps and audio-conditioning count. This [preregistered ablation](../superpowers/specs/2026-09-25-two-step-audio-conditioning-design.md) holds the two-step timesteps `(1000, 833.33333333, 0)` fixed and changes only the first DiT forward from `skip_audio=True` to `False`, giving two audio-conditioned calls. The new `--audio_first_step` flag is opt-in and valid only with `--denoising_steps 2`; defaults preserve the stock paths.

## Setup and smoke test

The development case is image 2 (SHA-256 `7f9c1d2567ca391200b1a76b2e07442111d4472493411b59195f80872d078550`), prompt `一个人在说话`, seed 43, exact 30-second 16-kHz PCM audio (SHA-256 `85372c99f11ca8396d9c01bb1aa1314bb165e93000938ba87f8ad06de104637e`), 416×720/24 FPS, FP8 GEMM/KV, CPU block/cache offload, `resident_kv_steps=0`, no compile/pin/anchor, offline FP8/T5 caches and `--disable_cudnn_benchmark`. Its controls are the archived [three-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-three.mp4) and [stock two-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-two.mp4) videos, generated with the same policy. The opt-in changes only the per-forward audio-skip flags; audio CFG remains at its default.

A [five-second input smoke video](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-smoke.mp4) completed with 117 decoded frames, a 4.875-second video stream and a 4.864-second audio stream. This is the generator's block-tail truncation of the 5.000-second PCM input, not five seconds of decoded output. Sampled peak GPU memory was 8,864 MiB and minimum machine-wide MemAvailable was 37,156 MiB. The [sparse contact sheet](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-smoke-contact.png) shows no gross face/hand failure at sampled seconds, which justifies the full-length gate but does not establish lip quality.

## Development-seed quality gate

The written gate requires ≥20% later-block time reduction versus three steps; SyncNet confidence no more than 0.30 below three steps **and** at least 0.35 above stock two steps on matched 747-window support with stable offset; ≥90% boundary motion retention, ≤125% mean boundary peak; ≥90% single-face coverage and reference-face cosine drop ≤0.03; and no conspicuous new visual defect in seven timepoints or five worst windows. Only a full pass authorizes the same candidate on seed 44. Otherwise the candidate is rejected without tuning timesteps or thresholds on image 2.

The new [seed-43 video](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-30s.mp4) contains 722 decoded frames and exactly 30.000 seconds of audio (SHA-256 `3e40dc806104dce85d4af62d3e9464d50c0b984bd3bbcc3b8d6cc05f349cfd95`). All three arms use the same image, PCM and seed, with cuDNN benchmarking disabled. The new arm's sampled peak GPU memory was 8,984 MiB, and its minimum machine-wide MemAvailable was 32,845 MiB.

| Seed-43 measure | Three steps | Stock two steps | Two steps, both calls audio-conditioned | Decision for new arm |
|---|---:|---:|---:|---|
| Median later-block time, 23 blocks | 21.439 s | 14.952 s | **15.291 s** | 28.7% below three steps; passes ≥20% |
| Sum of generation block times | 490.233 s | 341.834 s | 349.825 s | 2.064 block-only generated FPS; excludes setup/encoding |
| Mean boundary motion sum, 22 windows | 31.741 | 36.001 | **32.015** | 100.9% of baseline; passes ≥90% |
| Mean boundary-window peak | 6.158 | 6.591 | **5.775** | 93.8% of baseline; passes ≤125% |
| Single-face detections | 31/31 | 31/31 | **31/31** | Passes ≥90% |
| Mean reference-face cosine | 0.698 | 0.729 | **0.715** | No decline versus baseline; pose-sensitive proxy |
| SyncNet confidence, matched 747×31 support | 6.938 | 6.271 | **6.868** | −0.070 vs three, +0.597 vs stock two; passes both gates |
| SyncNet best offset at converted 25 FPS | −2 | −2 | **−2** | Stable |

The [three-arm seven-timepoint contact sheet](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-contact.png) and [new arm's five worst boundary-window sheet](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-worst5.png) show no conspicuous new hand/face smear in sampled frames. These stills cannot establish gesture naturalness or speech articulation. The new [motion](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-motion.json), [face](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-face.json) and [SyncNet score](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed43-syncnet.json) records, distance matrix, command/resource JSONs and compressed logs are archived alongside the video. Seed 43 passes every predeclared gate and therefore triggers the seed-44 replication; it is not sufficient alone for an upstream quality claim.

## Seed-44 replication and decision

Seed 44 used the same image, exact PCM, prompt, resolution, FP8/offload configuration and cuDNN control, changing only the noise seed from the development run. The new [seed-44 video](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-30s.mp4) contains 722 frames and 30.000 seconds of audio (SHA-256 `6dab1cf0a8c67ec68fe9fb47d1e0f48c64c2395b805a30edc233a93152ed4f0c`). Its SyncNet distance matrix is again **747×31**, and the best offset remains −2 frames. The [paired speed/lip/motion chart](artifacts/2026-09-25-two-step-audio-conditioning/two-step-first-audio-frontier.png) makes the trade-off visible.

| Seed-44 measure | Three steps | Stock two steps | Two steps, both calls audio-conditioned | Decision for new arm |
|---|---:|---:|---:|---|
| Median later-block time, 23 blocks | 21.467 s | 14.964 s | **15.211 s** | 29.1% below three steps; passes ≥20% |
| Sum of generation block times | 490.994 s | 342.401 s | 347.853 s | 2.076 block-only generated FPS; excludes setup/encoding |
| Mean boundary motion sum, 22 windows | 38.309 | 35.450 | **29.943** | **78.2% of baseline; fails ≥90%** |
| Mean boundary-window peak | 6.799 | 6.410 | **5.402** | 79.4% of baseline; passes ≤125% but does not rescue low motion |
| Single-face detections | 31/31 | 31/31 | **31/31** | Passes ≥90% |
| Mean reference-face cosine | 0.731 | 0.734 | **0.713** | −0.018; passes ≤0.03 decline, pose-sensitive proxy |
| SyncNet confidence, matched 747×31 support | 6.644 | 6.214 | **6.782** | +0.137 vs three, +0.567 vs stock two; passes lip gate |
| SyncNet best offset at converted 25 FPS | −2 | −2 | **−2** | Stable |

The [seed-44 contact sheet](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-contact.png) shows fewer/lower hand gestures in the new arm at some sampled times (notably around 20–25 seconds) than the three-step control, consistent with the measured local-motion deficit. The [five worst boundary windows](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-worst5.png) show no obvious new face/hand smear; lower grayscale peaks alone cannot distinguish improved continuity from weaker movement. The underlying [motion](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-motion.json), [face](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-face.json), [SyncNet score](artifacts/2026-09-25-two-step-audio-conditioning/liveact-two-step-audio-first-seed44-syncnet.json) and matrix are preserved.

**Reject the unmodified first-audio two-step policy as a quality-preserving default.** Adding audio to the first call recovered the SyncNet proxy on *both* seeds with only about 2% extra block time relative to stock two-step, which is evidence that this intervention can improve the lip metric at fixed timestep boundaries. But seed 44 retained only 78.2% of three-step boundary-window motion, below the preregistered 90% gate, and sampled gestures were visibly reduced. The candidate therefore fails the multi-objective gate; no held-out identity run or official SoulX PR is justified. The mechanism remains specific to this model, identity/audio and inference policy: a relative SyncNet score is not ground-truth speech articulation, and grayscale frame change is not a complete human-motion metric. A future study would need a separately specified conditioning schedule or objective that balances lip fidelity with expressive motion, then fresh-seed and held-out-identity checks. These runs remain around 2.06–2.08 block-only generated FPS, far from 24-FPS real time on this 4090.

**Post hoc motion-source audit:** [A fixed-region/flow/pose check](2026-09-25-spatial-motion-source-audit.md) confirms less seed-44 image change near chunk boundaries, but finds lower-body optical flow **higher outside** those windows and no uniform drop in tracked wrist travel. Thus the original gate failure remains, while “all body motion is suppressed” would overstate what the grayscale proxy proves. The observed smaller gestures in sparse frames are local examples, not an all-time summary.
