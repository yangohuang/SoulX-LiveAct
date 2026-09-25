# Long-horizon stability: second-seed replication ledger

This ledger follows the [predeclared protocol](2026-09-25-seed43-replication-protocol.md) and the [seed-42 pilot](2026-09-25-long-horizon-stability-study.md). Every pair uses seed 43, the same source image/audio and runtime configuration within the pair, 416×720 at 24 FPS, three denoising steps and 722 decoded frames. The methods are separate sampled runs, so their frames are not pixel-aligned counterfactuals. A temporary startup failure on the first image-1 baseline attempt was caused by the new worktree lacking its local checkpoint symlink; it produced no MP4. The link was restored to the same checkpoint files used by the pilot, and the successful baseline was rerun from the original request.

## Identity 1: complete

The [baseline video](artifacts/videos/image1-baseline-seed43-30s.mp4), [anchor video](artifacts/videos/image1-anchor045-seed43-30s.mp4) and [side-by-side review](artifacts/videos/image1-baseline-left-anchor045-right-seed43-30s.mp4) are available. The [motion JSON](artifacts/2026-09-25-image1-seed43-motion-pair.json), [face-reference JSON](artifacts/2026-09-25-face-identity-image1-seed43.json), [SyncNet JSON](artifacts/2026-09-25-image1-seed43-syncnet.json), [seven-timepoint contact sheet](artifacts/2026-09-25-image1-seed43-long-horizon-contact.png) and [worst-window frames 116–123](artifacts/2026-09-25-image1-seed43-window116-123.png) contain reproducible details and hashes.

| Metric | Baseline | Anchor 0.45 | Paired reading |
|---|---:|---:|---|
| Mean eight-transition boundary peak, grayscale MAE | 3.364 | 2.535 | −24.6% |
| Mean eight-transition change sum | 18.653 | 13.225 | −29.1%; motion suppression or altered trajectory remains possible |
| Largest boundary-window peak | 7.424 | 4.438 | Both at window 116; peak offset 2→4 |
| Windows with peak after seam | 12/22 | 16/22 | More delayed local maxima |
| Mean reference-face cosine, 31/31 single-face detections | 0.744 | 0.734 | No identity-proxy gain; pose-sensitive |
| SyncNet minimum distance | 8.062 | 8.260 | Higher is unfavorable |
| SyncNet relative confidence | 6.529 | 6.296 | Lower is unfavorable; best offset stays −2 converted frames |

The 25-FPS SyncNet pipeline produced 754-frame, 481,536-audio-sample face crops in both arms. The frame-116 sequence shows different hand timing between separately sampled runs; the anchor does not prove more natural motion just because its grayscale peak is lower. Identity 1 now has two seeds showing lower peak, lower local image change, an unfavorable SyncNet shift and no reference-face advantage. This is a replicated **metric trade-off**, not a validated causal effect on perception or a long-form identity-drift estimate.

## Identity 2: complete

The [baseline video](artifacts/videos/image2-baseline-seed43-30s.mp4), [anchor video](artifacts/videos/image2-anchor045-seed43-30s.mp4) and [side-by-side review](artifacts/videos/image2-baseline-left-anchor045-right-seed43-30s.mp4) have 722 frames each. The [motion JSON](artifacts/2026-09-25-image2-seed43-motion-pair.json), [face-reference JSON](artifacts/2026-09-25-face-identity-image2-seed43.json), [SyncNet JSON](artifacts/2026-09-25-image2-seed43-syncnet.json), [seven-timepoint contact sheet](artifacts/2026-09-25-image2-seed43-long-horizon-contact.png) and [frames 500–507](artifacts/2026-09-25-image2-seed43-window500-507.png) preserve the evidence.

| Metric | Baseline | Anchor 0.45 | Paired reading |
|---|---:|---:|---|
| Mean eight-transition boundary peak, grayscale MAE | 5.998 | 4.277 | −28.7% |
| Mean eight-transition change sum | 31.409 | 22.255 | −29.1% |
| Largest boundary-window peak | 14.823 | 8.231 | Both at window 500; peak offset 3→4 |
| Windows with peak after seam | 15/22 | 19/22 | More delayed local maxima |
| Mean reference-face cosine, 31/31 single-face detections | 0.688 | 0.674 | No identity-proxy gain; pose-sensitive |
| SyncNet minimum distance, matched 754-frame crops | 8.059 | 8.194 | Higher is unfavorable |
| SyncNet relative confidence, matched 754-frame crops | 6.955 | 6.668 | Lower is unfavorable; best offset stays −2 |

The baseline MP4 container reported 30.125 seconds despite its 30.083-second video stream and 30.000-second audio stream; the anchor container reported 30.084 seconds. The SyncNet pipeline produced 755 versus 754 face-crop frames. A stream-copy trim removed only the extra baseline crop tail frame before the score above; both compared crops have 754 frames and 481,536 audio samples. The original 722-frame videos and motion/face scores are unchanged. The [frame-500 sequence](artifacts/2026-09-25-image2-seed43-window500-507.png) shows the anchor arm's raised hands acquiring a conspicuous dark, smeared appearance around frames 503–505 while the baseline motion is sharper. Separate trajectories prevent a pixel-aligned causal claim, but this is a concrete visual failure and rejects a quality improvement claim. Identity 2 now shows the same lower-MAE/lower-motion/worse-lip pattern at both seeds.

## Remaining pair

Identity 3 at seed 43 is running sequentially on the same RTX 4090. It will be added only after both arms pass video, motion, face and lip-sync checks; partial or failed runs will be disclosed rather than silently dropped. The [input manifest](artifacts/2026-09-25-seed43-input-manifest.json) fixes reference and audio hashes for all three identities.
