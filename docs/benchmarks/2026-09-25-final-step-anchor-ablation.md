# Full-rollout skip-final velocity anchor: hand/lip recovery, motion gate failed

The [block-16 VAE replay](2026-09-25-hand-artifact-source-attribution.md) isolated the final denoising-step velocity-anchor correction as sufficient to create one white/dark hand smear under fixed latent history. The [prospective full-rollout design](../superpowers/specs/2026-09-25-final-step-anchor-ablation.md) asks whether omitting only that correction in *every later block* preserves the local hand/lip benefit across a full 30-second causal generation without suppressing gestures. This is an opt-in schedule ablation via `--motion_anchor_skip_last_step`, not a new model or a default fix. It performs the same three DiT denoising passes and skips one latent anchor arithmetic call per later block.

## Development setup and visual result

Identity 2, seed 43, public reference image 2, first 30 seconds of audio 2, prompt `一个人在说话`, 416×720, 24 FPS, three denoising steps, FP8 GEMM/KV, block/cache offload and one resident KV step on one RTX 4090. The candidate keeps velocity anchor strength 0.45 at steps 0 and 1, skips step 2. The original velocity arm applies it at all three steps; the baseline applies none. The [candidate MP4](artifacts/videos/image2-velocity-skip-final-seed43-30s.mp4) decodes to 722 frames, SHA-256 `71eba3d77f1e6fdf2aa3a6eecdd3e9646ad1e157c06621d47de586d9181a5534`. All three videos are separate rollouts, so their frames are not pixel-aligned counterfactuals.

The [frames 500–507 three-arm sheet](artifacts/2026-09-25-image2-skip-final-window500-three-row.png) and [hand crop](artifacts/2026-09-25-image2-skip-final-hands-three-row.png) show the candidate no longer has the all-step velocity arm's obvious white/dark hand trails. The candidate's [five highest boundary-peak windows](artifacts/2026-09-25-image2-skip-final-top5-windows.png), at transitions 500, 276, 436, 20 and 212, were reviewed; no comparable new smear was visible in these short eight-frame windows. The absence of a conspicuous artifact in this review is narrower than a full-video perceptual pass. Gestures near window 500 are also less expansive than the unanchored reference, consistent with the motion-sum evidence below.

| Measure | Unanchored baseline | All-step velocity 0.45 | Skip-final velocity 0.45 |
|---|---:|---:|---:|
| Mean 22-boundary immediate grayscale change | 4.114 | 4.291 | 3.401 |
| Mean eight-transition peak | 5.998 | 5.317 | 5.213 |
| Mean eight-transition motion sum | **31.409** | 26.561 | **26.468** |
| Peak after the immediate seam | 15/22 | 11/22 | 15/22 |
| Window 500 immediate change | 4.377 | 6.747 | 4.288 |
| Window 500 eight-transition peak | 14.823 | 10.375 | 10.668 |
| SyncNet relative confidence; matched crops | **6.955** | 6.224 | **6.950** |
| SyncNet minimum distance; lower favorable | 8.059 | 8.508 | **8.005** |
| Mean reference-face cosine; all 31/31 faces detected | 0.688 | 0.668 | 0.695 |

The [motion JSON](artifacts/2026-09-25-image2-skip-final-motion-three-way.json), [matched-crop SyncNet JSON](artifacts/2026-09-25-image2-skip-final-syncnet.json) and [face-proxy JSON](artifacts/2026-09-25-image2-skip-final-face-identity.json) provide hashes, full per-boundary values, detection coverage and the 31-point face curve. The candidate retains only **84.3%** of baseline mean window motion (`26.468 / 31.409`), below the prespecified **90%** gate. Its lower grayscale peak cannot be called better action quality. The face-reference cosine is sensitive to pose/expression and is not proof of identity improvement.

The local SyncNet S3FD pipeline converted all arms to 25 FPS. Candidate and previously scored baseline/velocity source muxes produce 755 cropped video frames despite 722-frame, 24-FPS video streams. Only the extra final converted crop frame was stream-copied away (`-t 30.16`), yielding the same 754 face frames and 481,536 decoded 16-kHz audio samples used for the earlier two scores. The SyncNet model SHA-256 is `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`; best offset is −2 converted frames in all arms. This is a proxy score, so visual lip review remains necessary before a quality claim. The 31 sampled frames in the local CPU InsightFace evaluation each contained one detected face, satisfying its coverage rule; the cosine is not a validated long-horizon drift measure.

## Cost and decision

The code-level DiT forward count, three denoising steps and KV strategy are unchanged. Omitting the final anchor saves one elementwise latent correction per later block; it does not make the 4090 real time. MP4 modification time minus [generation-log](artifacts/2026-09-25-image2-skip-final-generation.log) birth time was about 455 seconds for this candidate and about 455 seconds for the prior all-step velocity run, but separate-run cache and scheduling conditions make that an **uncontrolled** wall-time comparison. Source SHA-256: `generate.py` `a280673ef02ba302b1221fbb86a8aabb3344271c88134356b1d93f338193f08e`, `temporal_continuity.py` `7228d13e2a9428b66f3a8bbc2343a1807eaea910ddd3cc0443221e4a7c0f7da0`; reference image and audio-trim hashes are `7f9c1d2567ca391200b1a76b2e07442111d4472493411b59195f80872d078550` and `85372c99f11ca8396d9c01bb1aa1314bb165e93000938ba87f8ad06de104637e`.

**Decision: reject as a general quality improvement.** It clears the observed hand smear and restores the SyncNet proxy to near-baseline on this one development case, but fails the preregistered motion-retention gate by 5.7 percentage points. Per the study design, no held-out identity was generated and no further strength or schedule tuning was done. The option remains default-off for research; it is not a quality PR candidate, a 24-FPS result, or evidence of infinite-video stability. The transferable insight is narrower: a denoising-step intervention can improve one visible artifact and lip proxy while still shrinking action, so long-video evaluation must keep motion amplitude and timing alongside seam and sync metrics.
