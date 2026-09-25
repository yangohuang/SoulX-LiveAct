# LiveAct as a long-horizon stability testbed: 30-second pilot

## Research question

Can a streaming-video intervention reduce abrupt chunk motion without merely delaying or suppressing motion, weakening lip synchronization, or increasing compute? The research contribution here is a reproducible diagnostic and a multi-objective decision rule. Repairing the single frame-373 event is **not** the success definition. This is directly relevant to the long-form generation, error accumulation, temporal consistency and inference-efficiency directions in the [Vivix-W1 report](https://vivix.ai/tech-report-vivix-w1), but uses SoulX-LiveAct's public inference system rather than Vivix's model or methods.

## Fixed pilot

Three existing 30-second, 416×720, 24-FPS, 722-frame videos use the same documented image, audio, prompt, seed 42, three denoising steps and FP8/KV/offload setup: [baseline](artifacts/videos/baseline-30s.mp4), [anchor 0.25](artifacts/videos/anchor-025-30s.mp4), and [anchor 0.45](artifacts/videos/anchor-045-30s.mp4). Each is a separately sampled run. Capturing probes and other minor execution differences previously produced nonidentical frames even with the same seed, so these are matched-condition comparisons, **not** pixel-aligned counterfactuals. This is one identity and one seed, not a population-level effect estimate.

[`long_horizon_stability.py`](../../long_horizon_stability.py) uses FFmpeg `-vsync 0` to decode 180×104 grayscale frames and computes adjacent-frame mean absolute difference (MAE). The first chunk contributes 21 frames and each later chunk 32. Its 22 boundary transition indices are `20 + 32k`; every boundary is scored over its first eight transitions. The output preserves all 721 transitions and each window's immediate value, peak, peak offset and summed motion in [JSON](artifacts/2026-09-25-long-horizon-stability.json). All source MP4s and their SHA-256 hashes are included. [The timeline](artifacts/2026-09-25-long-horizon-stability.png) highlights the largest pose event.

```bash
/home/yg/miniforge3/envs/digithuman/bin/python long_horizon_stability.py \
  --video baseline docs/benchmarks/artifacts/videos/baseline-30s.mp4 \
  --video anchor025 docs/benchmarks/artifacts/videos/anchor-025-30s.mp4 \
  --video anchor045 docs/benchmarks/artifacts/videos/anchor-045-30s.mp4 \
  --output docs/benchmarks/artifacts/2026-09-25-long-horizon-stability.json
```

## Measured movement and independent guardrails

MAE is on an 0–255 grayscale scale; summed motion is the total of eight MAEs, **not** a physical movement distance. Lower is not automatically better.

| Arm | Mean immediate change | Mean eight-transition peak | Largest peak | Mean eight-transition sum | Windows with delayed peak |
|---|---:|---:|---:|---:|---:|
| Baseline | 2.688 | 4.130 | 15.756 | 17.943 | 12/22 |
| Anchor 0.25 | 1.955 | 3.045 | 10.368 | 14.565 | 12/22 |
| Anchor 0.45 | 1.741 | 2.721 | 5.903 | 12.689 | 14/22 |

Relative to baseline, anchor 0.25 reduces the mean window peak by 26.3% and summed window motion by 18.8%; anchor 0.45 reduces them by 34.1% and 29.3%. At the visually salient window starting at transition 372, all three peak one transition later: 15.756, 10.368 and 5.903. Thus this fixed-anchor comparison does **not** show a newly delayed large spike within those eight transitions. A different low-pass anchor experiment previously moved the hand lift from around transition 373 to 376; that example is documented in the [boundary probe](2026-09-24-liveact-boundary-probe.md) and is why the peak offset is retained instead of reporting seam MAE alone.

The lower summed window motion also raises a **motion-suppression or trajectory-change hypothesis**. The average change *outside* these boundary windows is 1.642, 1.640 and 1.687 for baseline, 0.25 and 0.45, respectively. That does not indicate global freezing, but it cannot establish that the hand completed the same natural motion. The [existing side-by-side 30-second review](artifacts/videos/baseline-left-anchor045-right-30s.mp4) and key frames must be judged for gesture timing and plausibility; frame MAE alone cannot do that. Early/middle/late mean window peaks are 4.151/4.643/3.594 for baseline, 3.247/3.477/2.384 for 0.25, and 3.127/2.690/2.287 for 0.45. A single 30-second sample cannot estimate an hour-scale error slope.

The previous [same-pipeline SyncNet evaluation](2026-09-24-liveact-motion-continuity.md) gives baseline/0.25/0.45 minimum embedding distance **7.672/7.657/7.989** (lower preferred) and relative confidence **7.079/7.012/6.599** (higher preferred). The stronger anchor therefore has an unfavorable lip-sync signal even as the motion-change proxy falls. These are relative scores after 25-FPS conversion, not absolute lip delay or a perceptual listening test. Existing inference logs put later chunks near 19 seconds for 32 output frames for all three arms, so none is real-time on this 4090; these are separate runs rather than a kernel benchmark.

### Face-reference consistency is a separate, noisy proxy

[`face_identity_stability.py`](../../face_identity_stability.py) uses a local InsightFace buffalo_l detector and recognition model on CPU, sampling frame indices 0, 24, …, 720 (31 samples). Its weights are **not** included in this repository, and no biometric embeddings are saved. The [31-point JSON](artifacts/2026-09-25-face-identity-image1.json) records SHA-256 hashes for the reference, both model files and source MP4s; the [timeline](artifacts/2026-09-25-face-identity-image1.png) plots cosine similarity to the supplied reference image. Frames with zero or multiple detected faces are excluded and counted, and an early/late coverage below 90% blocks interpretation. Here every arm has exactly one detected face in all 31 samples, with minimum detection score 0.777, so the first-generated-face anchor is frame 0. The detector sometimes places the left edge of the face box outside the image, which is another reason to treat this as a proxy.

| Arm | Mean reference cosine | 10th percentile | First-third mean | Last-third mean | Final sample |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.734 | 0.624 | 0.761 | 0.705 | 0.777 |
| Anchor 0.25 | 0.728 | 0.636 | 0.768 | 0.695 | 0.786 |
| Anchor 0.45 | 0.724 | 0.624 | 0.763 | 0.691 | 0.781 |

All three traces have large second-to-second excursions and rebound near the final sample. The lower last-third means do **not** demonstrate permanent identity drift, and no anchor has a clear identity advantage on one identity/seed. Expression, pose and crop changes can affect face embeddings. This is now an instrumented identity-consistency pilot, not a validated long-video identity benchmark.

```bash
OMP_NUM_THREADS=1 /home/yg/miniforge3/envs/latentsync/bin/python face_identity_stability.py \
  --reference examples/image/1.png \
  --video baseline docs/benchmarks/artifacts/videos/baseline-30s.mp4 \
  --video anchor025 docs/benchmarks/artifacts/videos/anchor-025-30s.mp4 \
  --video anchor045 docs/benchmarks/artifacts/videos/anchor-045-30s.mp4 \
  --output docs/benchmarks/artifacts/2026-09-25-face-identity-image1.json
```

A CPU MediaPipe Pose spot check at frames 0, 100, 250, 370–378, 400, 600 and 720 could not validate hand kinematics: both wrist landmarks were largely outside the visible frame (`y > 1` or `x < 0`) and wrist visibility was about 0.01–0.07. Those coordinates are excluded from the score rather than interpreted as physical motion. Gesture timing therefore still needs visual review or a fit-for-purpose arm/hand tracker on suitable framing.

## What this study establishes

It establishes a reproducible way to reject a misleading one-number claim: temporal stability must consider delayed spikes, total motion, visual action timing and independent lip-sync/identity checks. On this single sample, anchors lower several image-change metrics, but the stronger option does not pass a multi-objective quality gate because its lip-sync proxy worsens and natural gesture quality is unresolved. The face-reference curve supplies no clear identity advantage. It does **not** establish a general algorithmic improvement, physical consistency, long-form identity preservation, or a LiveAct model repair.

## Cross-identity extension protocol, fixed before reading new outputs

The first extension runs images/audio 2 and 3, each trimmed to 30 seconds, with seed 42 and two arms (baseline versus fixed anchor 0.45). The same 416×720, 24-FPS, three-step FP8/one-resident-KV setup and the evaluator above are fixed. The primary descriptive endpoint is mean eight-transition peak at all 22 boundaries; secondary endpoints are maximum peak, eight-transition summed motion and early/middle/late values. A lower peak in both new identities would support a reproducible *image-change* effect, but would still not establish better motion, identity, lip sync or generality across seeds. Each generated MP4 must have 722 frames and decode successfully; report failures and do not silently omit a run. The full six-pair/two-seed gate below remains necessary for a quality claim.

The next replication gate is at least three identities × two seeds × 30 seconds, with fixed audio, prompt, model and runtime settings per identity. Each arm needs the same source-specific 22-boundary analysis, manual review of its worst three windows, a validated face-embedding drift curve, and comparable SyncNet extraction. This separation follows the spirit of [VBench's temporal-quality dimensions](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf), especially subject consistency, motion smoothness and dynamic degree; the present grayscale metric is not a VBench score. Only then should a new inference method be compared with the baseline. A separate model-side direction would study bounded history or reference refresh under the same evaluation protocol; generated-history training should wait for image/audio-aligned conditioning and a valid distributional target. A negative or mixed result remains a useful research outcome and need not become an upstream PR.
