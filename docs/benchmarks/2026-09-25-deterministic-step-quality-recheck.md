# Two versus three steps with deterministic conditioning on one RTX 4090

The [first two-step frontier](2026-09-25-two-step-quality-frontier.md) was fast but missed its lip-sync gate. A subsequent [fixed-seed audit](2026-09-25-fixed-seed-determinism-diagnostic.md) found that default cuDNN benchmarking could change Wav2Vec and reference-VAE tensors between independent processes even when the input and seed were identical. This experiment repeats the schedule comparison with `--disable_cudnn_benchmark` in both arms, following the [frozen decision rule](../superpowers/specs/2026-09-25-deterministic-step-quality-recheck.md).

## Controlled seed-43 pair

Both arms used image 2, the archived exact 30-second PCM WAV, prompt `一个人在说话`, seed 43, 416×720 at 24 FPS, the same FP8 GEMM/KV and CPU cache/block-offload configuration, `resident_kv_steps=0`, offline FP8/T5 caches, and no compile, pinning or latent anchor. Only the number of diffusion steps changed. Both outputs contain 722 decoded frames and 30.000 seconds of audio. Generation command, request and machine-wide memory samples are preserved with each arm. The `--disable_cudnn_benchmark` flag is an experimentally verified same-machine reproducibility control, not a cross-version determinism guarantee.

| Predeclared measure | Three steps | Two steps | Gate |
|---|---:|---:|---|
| Median later-block time, 23 blocks | 21.439 s | 14.952 s | 30.3% lower, passes ≥20% |
| Sum of generation block times | 490.233 s | 341.834 s | 1.472→2.112 generated FPS, excludes setup/encoding |
| Mean eight-transition boundary motion sum, 22 boundaries | 31.741 | 36.001 | 113.4% retained, passes ≥90% |
| Mean boundary-window peak | 6.158 | 6.591 | 107.0%, passes ≤125% |
| Single-face coverage, one sample/s | 31/31 | 31/31 | Both pass ≥90% |
| Mean reference-face cosine | 0.698 | 0.729 | No decline; pose/expression-sensitive proxy |
| SyncNet confidence, identical 747×31 scored support | **6.938** | **6.271** | **−0.667, fails ≤0.30 loss** |
| SyncNet best offset at converted 25 FPS | −2 frames | −2 frames | Stable |

The [three-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-three.mp4) and [two-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-two.mp4) videos have SHA-256 `5124efdbcb70ac10f2ffdeaa50e7cac4a8d7f010c41c4b6da38b0102f1b34645` and `6a2e7cb4aea46d7a0bdd72bb00c1039fd7ead1d73d8d82ef044913fc2c86c9bf`. The [seven-timepoint sheet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-contact.png) and [three-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-three-worst5.png) / [two-step](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-two-worst5.png) five-worst-boundary sheets show no new gross face/hand smear in sampled frames. These still images do not verify gesture naturalness or lip accuracy. SyncNet is a proxy, but the identical support and shared best offset rule out a simple score-window or global timing shift explanation for this gate failure. Five-second exploratory bins show losses in four of six bins, with gains in the first and fifth; this is not a formal temporal-drift test.

The [motion](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-motion.json), [face](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-face.json), [three-step SyncNet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-three-syncnet.json) and [two-step SyncNet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed43-two-syncnet.json) JSONs preserve all measurements. The compressed distance matrices and original generation logs are in the same artifact directory.

## Preregistered seed-44 replication

Seed 44 changes only the noise seed while retaining the same image, PCM, prompt, hardware, precision/offload policy and deterministic cuDNN flag. Both fresh outputs again contain 722 decoded frames and 30.000 seconds of audio. The two SyncNet matrices again each have **747×31** audio-supported windows, with the same model and best offset of −2 frames. The [paired measurement chart](artifacts/2026-09-25-deterministic-step-recheck/seed-paired-step-frontier.png) shows all predeclared quantitative gates by seed.

| Predeclared measure | Three steps | Two steps | Gate |
|---|---:|---:|---|
| Median later-block time, 23 blocks | 21.467 s | 14.964 s | 30.3% lower, passes ≥20% |
| Sum of generation block times | 490.994 s | 342.401 s | 1.471→2.109 generated FPS, excludes setup/encoding |
| Mean eight-transition boundary motion sum, 22 boundaries | 38.309 | 35.450 | 92.5% retained, passes ≥90% |
| Mean boundary-window peak | 6.799 | 6.410 | 94.3%, passes ≤125% |
| Single-face coverage, one sample/s | 31/31 | 31/31 | Both pass ≥90% |
| Mean reference-face cosine | 0.731 | 0.734 | No decline; pose/expression-sensitive proxy |
| SyncNet confidence, identical 747×31 scored support | **6.644** | **6.214** | **−0.430, fails ≤0.30 loss** |
| SyncNet best offset at converted 25 FPS | −2 frames | −2 frames | Stable |

The [three-step video](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-three.mp4) SHA-256 is `bb2dfe0c47db4e6a02f27db72cdbd7076d99c7959c4a79aa9c8efe95bddea6b0`; the [two-step video](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-two.mp4) is `59cff4fccbc580c4deebfa45c38c931b0b42bff6c453130af2b9c124e711bb13`. The [seven-timepoint sheet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-contact.png), [three-step worst-five](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-three-worst5.png) and [two-step worst-five](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-two-worst5.png) show no conspicuous new smear at sampled frames. The [motion](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-motion.json), [face](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-face.json), [three-step SyncNet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-three-syncnet.json) and [two-step SyncNet](artifacts/2026-09-25-deterministic-step-recheck/liveact-deterministic-steps-image2-seed44-two-syncnet.json) records allow rechecking the numbers.

## Decision rule and scope

**Reject the existing two-step schedule as a quality-preserving default.** Both deterministic, matched seeds pass speed, motion and face-proxy gates but fail the predeclared SyncNet loss gate by 0.667 and 0.430, respectively. This makes the *observed trade-off for this intervention on this image/audio* reproducible under controlled cuDNN conditions. The 3→2 step change removes the middle audio-conditioned DiT call as well as changing the timestep integration; this experiment cannot separate those mechanisms. It is not evidence that all two-step schedules, other identities, or other audio content fail. The faster arm's block-only 2.109–2.112 generated FPS remains over 11× slower than 24-FPS live output, before setup/encoding. A follow-up could restore audio conditioning on the first of two calls and test whether lip confidence returns at roughly the same compute, then require held-out identities before an upstream quality claim. That is a new hypothesis, not a result of this experiment.
