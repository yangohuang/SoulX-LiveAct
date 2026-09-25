# Velocity-aware latent anchor: controlled negative ablation

## Hypothesis and implementation

The [six paired cases](2026-09-25-seed43-replication.md) reject the fixed `0.45` latent anchor as a general quality improvement: it reduces boundary pixel differences but also removes local motion, worsens SyncNet confidence and smears hands. The [predeclared alternative](../superpowers/specs/2026-09-25-velocity-aware-boundary-design.md) asks whether pulling the first new clean latent toward `last + (last - penultimate)`, rather than toward `last`, better preserves a continuing gesture. The second new latent targets `last + 2*(last - penultimate)` at one third the strength. This is only first-order **latent extrapolation**, not physical motion estimation, model retraining or Vidu Self-Replay Forcing.

The opt-in `--motion_anchor_mode velocity` is implemented in [`temporal_continuity.py`](../../temporal_continuity.py) and [`generate.py`](../../generate.py); the default `hold` mode preserves the prior behavior. The three-step generation path and number of DiT forwards are unchanged. The candidate uses additional elementwise latent arithmetic, which was not isolated in a kernel-level latency benchmark. The complete LiveAct test suite passed 55 tests after the change. For the run below, `generate.py` and `temporal_continuity.py` had SHA-256 `c96f5836a8d02544c9ba5c97005c286561859ef66646a76e3d3bd2f1789c0d85` and `007db549094470a587b28934ff1cd999ccf06c70efaa82688e4090ee993f2747`, respectively.

## Fixed development case

Identity 2, seed 43, first 30 seconds of example audio 2, reference image 2, prompt `一个人在说话`, 416×720, 24 FPS, three denoising steps and the same FP8/KV/offload settings were held fixed. The [baseline](artifacts/videos/image2-baseline-seed43-30s.mp4), [fixed anchor](artifacts/videos/image2-anchor045-seed43-30s.mp4) and [velocity anchor](artifacts/videos/image2-velocity045-seed43-30s.mp4) each decode to 722 frames. A [three-way review video](artifacts/videos/image2-baseline-hold-velocity-seed43-30s.mp4) is ordered baseline/hold/velocity from left to right. The [motion JSON](artifacts/2026-09-25-image2-velocity-anchor-ab.json), [face JSON](artifacts/2026-09-25-face-identity-image2-velocity045-seed43.json), [matched-crop SyncNet JSON](artifacts/2026-09-25-image2-velocity-syncnet.json), [seven-timepoint contact sheet](artifacts/2026-09-25-image2-velocity-long-horizon-contact.png) and [frames 500–507](artifacts/2026-09-25-image2-velocity-window500-507.png) preserve the evidence.

| Measure | Baseline | Hold 0.45 | Velocity 0.45 |
|---|---:|---:|---:|
| Mean eight-transition boundary peak, grayscale MAE | 5.998 | 4.277 | 5.317 |
| Mean eight-transition change sum | 31.409 | 22.255 | 26.561 |
| Largest boundary-window peak | 14.823 | 8.231 | 10.375 |
| Windows with peak after seam | 15/22 | 19/22 | 11/22 |
| Window 500 immediate change | 4.377 | 3.497 | 6.747 |
| Mean reference-face cosine; 31/31 detected | 0.688 | 0.674 | 0.668 |
| SyncNet minimum distance; matched 754-frame face crops | 8.059 | 8.194 | 8.508 |
| SyncNet relative confidence; best offset −2 in all arms | 6.955 | 6.668 | 6.224 |

Relative to baseline, the velocity mode reduces the mean peak by 11.4% and window change sum by 15.4%, less than the hold anchor's 28.7% and 29.1%. A smaller motion reduction is not by itself better action. At the preselected worst baseline window 500, the velocity mode's immediate change is **higher** than baseline and its hands show bright white and dark trailing artifacts around frames 501–507. Its SyncNet relative score and pose-sensitive face-reference proxy are worse than both comparators. The separate sampling runs are not pixel-aligned counterfactuals, but the visible artifact is enough to fail the visual gate.

The baseline and velocity MP4 containers report 30.125 seconds despite 30.083-second video streams and 30.000-second audio streams. The SyncNet pipeline therefore made 755 face-crop frames for those arms versus 754 for hold. Only the extra final converted crop frame was removed by FFmpeg stream-copy; all three scores above use 754 video frames and 481,536 audio samples, while source videos and motion/face scores remain unmodified. The full source and matched-crop hashes are in the SyncNet JSON.

End-to-end wall times estimated from each log's first timestamp to the MP4 modification time were about 477, 471 and 454 seconds for baseline, hold and velocity, respectively. Different cache warmth and concurrent CPU scoring make these **unsuitable** as a controlled speed comparison; the code-level compute distinction is no extra DiT pass. This 4090 setup remains far from 24 FPS real-time.

**Decision: reject velocity anchoring at this strength and stop before the held-out identity-3 run**, as specified by the visual/lip gate. The failed case suggests that extrapolating compressed latents can amplify transient appearance errors; that mechanism is an inference from the observed white/dark hand trails, not a proven causal diagnosis. A future method needs explicit action or reference conditioning, or a selective fallback whose trigger and cost are measured on held-out sequences. Do not present this heuristic as a LiveAct fix or an upstream PR candidate.
