# Causal long-rollout baseline on one RTX 4090

This study asks whether unmodified, unanchored LiveAct shows a reproducible *duration-linked* deterioration in movement boundaries, face consistency or lip synchronization. It is a long-horizon measurement exercise for streaming digital-human research, not a real-time or infinite-video claim. The [prospective design](../superpowers/specs/2026-09-25-90s-causal-stability-baseline.md) fixed the decision rule before the 90-second output was measured; a separate [archived 30-second calibration](artifacts/2026-09-25-archived-30s-trend-calibration.json) removed an overly sensitive event-count gate before this output existed.

## Frozen setup

One RTX 4090, 416×720, 24 FPS, three denoising steps, FP8 GEMM/KV, block/cache offload, one resident KV step, prompt `一个人在说话`, no latent anchor, no probes. The [input manifest](artifacts/2026-09-25-causal-long-rollout-input-manifest.json) records references, source/trim hashes and output hashes. Identity 2 uses `examples/image/2.png` and the first 90.000 seconds of `examples/audio/2.wav`, seed 43. The source audio was trimmed with `ffmpeg -i examples/audio/2.wav -t 90 -acodec pcm_s16le -ar 16000`; its SHA-256 is `3c55ef5fd1c096a2abc80c978cc5fa627344d4e8fa17c1b8a2c0fcd9c92a5e00`. Source image SHA-256 is `7f9c1d2567ca391200b1a76b2e07442111d4472493411b59195f80872d078550`. Generation uses the command family in the [rollout bridge record](2026-09-24-liveact-rollout-bridge.md), with `--seed 43`, the 90-second request JSON and no `--rollout_latent_dir` or anchor flags.

The [playable 90-second video](artifacts/videos/image2-baseline-seed43-90s.mp4) has 2162 decoded 24-FPS frames, a 90.083-second video stream and a 90.000-second audio stream. Its SHA-256 is `5464eb4f3b6d8066fac9dbc709062b4dcc158d90bcc2558983b9597906667bd8`. The [runtime record](artifacts/2026-09-25-image2-seed43-90s-runtime.json) has 68 generated blocks, median block cost 19.043 seconds, sum of block costs 1294.9 seconds and 118 GPU-memory samples with a maximum **16,276 MiB**. The generated-frame throughput calculated only from summed block time is **1.67 FPS**; startup and encoding make end-to-end throughput lower. This is offline generation on a 4090, not interactive real time.

## Ninety-second identity-2 result

The [motion JSON](artifacts/2026-09-25-image2-seed43-90s-motion.json) holds every decoded-frame transition and all 67 complete block windows. The [trend JSON](artifacts/2026-09-25-image2-seed43-90s-trend.json), [time chart](artifacts/2026-09-25-image2-seed43-90s-trend.png), [seven-timepoint contact sheet](artifacts/2026-09-25-image2-seed43-90s-contact.png) and [four high-peak eight-frame windows](artifacts/2026-09-25-image2-seed43-90s-worst-windows.png) support visual inspection.

| Fixed segment | Complete windows | Median eight-transition peak | Mean eight-transition motion sum | Peaks above first-segment P95 |
|---|---:|---:|---:|---:|
| 0–30 s | 22 | 5.133 | 32.175 | 2 |
| 30–60 s | 23 | 5.928 | 36.480 | 1 |
| 60–90 s | 22 | 5.256 | 34.961 | 3 |

The last/first median peak ratio is **1.024**, far below the preregistered 1.25 screen, while the motion-sum ratio is **1.087** and passes its 0.90 retention condition. High peaks occur in all thirds. In the reviewed windows at frames 500, 1556, 1876 and 1940, the large grayscale peaks correspond mainly to visible hand/face movement, and no obvious *new* late-third hand-trail failure appears in those particular eight-frame sheets. This sparse review does not certify all 90 seconds or fine hand anatomy.

The [face-proxy JSON](artifacts/2026-09-25-image2-seed43-90s-face.json) detects one face in all 91 once-per-second samples. Equal 30-sample bins at seconds 0–29/30–59/60–89 have mean reference-face cosine 0.6947/0.6956/0.6898 and mean first-face cosine 0.6607/0.6538/0.6433. The late–early changes, −0.0049 and −0.0174, are smaller than the preregistered 0.05 screen. Reference and first-frame similarity move with expression/pose and are not validated identity-drift estimators.

The [SyncNet record](artifacts/2026-09-25-image2-seed43-90s-syncnet.json) uses one continuous 25-FPS S3FD face track and recomputes each third's score from the saved [31-shift distance matrix](artifacts/2026-09-25-image2-seed43-90s-syncnet-distances.npz) with `score_syncnet_horizon.py`. Full-video confidence is 6.638; the three 30-second scores are **7.122 / 6.503 / 6.413**, with a stable −2-frame best offset. The last crop segment has 748 scored windows versus 750 in each earlier segment because SyncNet requires future video/audio context. Source-audio 20-ms RMS-active fractions are similar across thirds (0.875/0.855/0.865 at the recorded 0.01 threshold), but audio content and phonemes still vary. The declining SyncNet confidence is a *lip-proxy signal to investigate*, not proof of cumulative generation error or worse perceived lip sync. The crop has 2257 converted frames and 1,442,304 decoded 16-kHz audio samples; the score uses their common support.

## Sixty-second second-identity check

Identity 3 uses the first 60.000 seconds of its source WAV, seed 43 and the same inference settings. Its [playable video](artifacts/videos/image3-baseline-seed43-60s.mp4) has 1442 decoded 24-FPS frames; SHA-256 `0be0735cd28017e2612143c9fa8eada2253d514efbd02f547bd05bca5a1ce328`. The [motion](artifacts/2026-09-25-image3-seed43-60s-motion.json), [fixed-segment trend](artifacts/2026-09-25-image3-seed43-60s-trend.json), [face proxy](artifacts/2026-09-25-image3-seed43-60s-face.json), [time chart](artifacts/2026-09-25-image3-seed43-60s-trend.png), [five-timepoint contact sheet](artifacts/2026-09-25-image3-seed43-60s-contact.png) and [three high-peak windows](artifacts/2026-09-25-image3-seed43-60s-worst-windows.png) are preserved. All 61 once-per-second face samples contain one detected face.

| Fixed segment | Complete windows | Median eight-transition peak | Mean eight-transition motion sum | Mean reference/first-face cosine |
|---|---:|---:|---:|---:|
| 0–30 s | 22 | 6.017 | 33.213 | 0.7462 / 0.7378 |
| 30–60 s | 23 | 6.112 | 35.791 | 0.7280 / 0.7227 |

The late/early median peak ratio is **1.016**, motion-sum ratio **1.078**, and equal 30-sample-bin face changes are −0.0182 / −0.0151. The movement and face-proxy screens do not trigger. Contact samples preserve the same recognizable person, guitar, microphone and studio; the reviewed frame-116, frame-724 and frame-1428 windows show ordinary mouth/strumming changes rather than an obvious novel late failure. They cannot certify guitar-finger correctness. The [runtime record](artifacts/2026-09-25-image3-seed43-60s-runtime.json) gives 46 generated blocks, median block cost 19.093 seconds, sampled peak 16,316 MiB and block-time-only throughput 1.64 FPS. CPU evaluation overlapped part of this run, so this is not a controlled cross-identity speed comparison.

The second [SyncNet record](artifacts/2026-09-25-image3-seed43-60s-syncnet.json) and [distance matrix](artifacts/2026-09-25-image3-seed43-60s-syncnet-distances.npz) give full-video confidence 5.680 and **5.779 / 5.619** in the first/last 30 seconds. The best offset stays −1 converted frame; 750 and 748 windows are scored. The source audio's mono-mixed RMS rises from 0.069 to 0.135, and its 20-ms active fraction from 0.956 to 1.000, so this is a poor content-matched test of lip drift. This clip includes singing and guitar; SyncNet's proxy score should not be treated as a validated musical-performance metric. The continuous crop has 1505 converted frames and 961,920 decoded 16-kHz audio samples.

## Decision boundary

Neither observed trajectory triggers the preregistered late movement or face-proxy deterioration screen. Both lip-proxy scores decline, but by different amounts and under different speech/music content. The appropriate next controlled test is to repeat **the same source speech segment** at early and late times under a causal rollout and compare matched content, ideally with a fresh-history control, before attributing a trend to generated-history accumulation. These are two independent, unequal-duration, single-seed trajectories. No evidence here establishes infinite-horizon consistency, a forcing mechanism, or a new quality improvement. This study is a measurement artifact and interview case study, not an upstream quality-fix PR candidate.
