# Source attribution at the known frame-500 hand artifact

## Question

The velocity latent anchor in identity 2, seed 43, produces bright/dark hand trails near output frames 501–507. The pre-output scalar signal did not reliably distinguish that visual failure from a clear raised-hand gesture. This study asks whether the visible smear is already present after the final denoising correction, or is introduced mainly by VAE history/context decoding. It is a targeted causal probe, not a long-video quality improvement.

## Fixed experiment

Rerun exactly the 30-second identity-2 seed-43 unanchored and velocity-strength-0.45 arms at 416×720, 24 FPS, three denoising steps, same FP8/offload settings. Capture zero-based generated block 16, which starts at frame 501 and follows transition 500. Existing `--boundary_probe_block 16` saves the final current and prior-five clean latents, plus the first two predicted latents at each step. Add one opt-in file containing the *full* predicted current latents immediately **before** applying the final-step anchor, paired with the same prior five latents. Keep the default generation path unchanged.

Use the exact same VAE weights and decode call as `generate.py`: prior-last-three + current-eight, discard first nine decoded frames. First validate that same-arm final-latent re-decode approximately reconstructs output frames 501–508; require mean RGB MAE at most 8/255 after MP4 decoding and no conspicuous visual mismatch, or stop the source-attribution claim. The primary intervention is within the velocity run: same prior latents with final-step pre-anchor versus post-anchor current latents. Because earlier denoising steps and previous generated chunks were already anchored, this isolates only the *last* correction. Record before/after latent delta, RGB seam/first-eight-frame changes, hand appearance and VAE decode time. The secondary 2×2 matrix pairs prior history and current latents from the unanchored/velocity arms. Swapped combinations break each trajectory's normal alignment, so any new artifact is only evidence of VAE sensitivity, not a valid reconstruction or a recommended inference setting.

Gate: if the problem is not reproduced or same-arm replay does not match the captured output closely enough for visual comparison, do not claim source attribution. A lower pixel difference does not establish better hands; report visual rows and limitations. No default method change or official PR follows from a diagnostic-only result.
