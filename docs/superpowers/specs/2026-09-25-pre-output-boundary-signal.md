# Pre-output boundary signal: prospective diagnostic design

## Research question

Can a cheap scalar measured from an already-computed clean-latent prediction identify a risky chunk boundary *before decoding* without merely ranking legitimate large gestures? The earlier fixed and velocity latent anchors were rejected on visual/lip quality. This signal is a diagnostic first, not a corrective method or a claim of long-video stability.

## Measurement

For each generated block after block 0, record three float32 mean-absolute-error scalars at the first and last denoising steps, before any optional anchor: `gap = |predicted_first - previous_last|`, `previous_speed = |previous_last - previous_penultimate|`, and `velocity_residual = |(predicted_first - previous_last) - (previous_last - previous_penultimate)|`. Average over channels and spatial positions. These are means of absolute differences, not absolute differences of means. Each record includes zero-based block/step indices, timestep, and `block_start_frame`; the corresponding decoded boundary is transition `(block_start_frame-1, block_start_frame)`. Keep diagnostics off by default. Do not save full latents or alter sampling/RNG. Export per-request JSON after generation with input paths, seed and settings.

## Prospective evaluation

Run the unanchored identity-2 example for 30 seconds at 416×720, 24 FPS, three steps and seeds 42 and 43. Also rerun the seed-43 velocity-anchor arm as a *known visual-failure control*: previous review found conspicuous hand smearing near frames 501–507. Compare each arm's 22 boundaries with the adjacent-frame grayscale-MAE series of the *same generated video*. Report Spearman rank association and top-five overlap for `gap` and `velocity_residual` against the first-eight-transition peak. Inspect at least the five highest final-step signal windows and five highest pixel-peak windows per seed, plus the known failure window; distinguish ordinary hand/head motion, apparent ghosting, timing discontinuity and ambiguous cases. A large pixel peak alone is not an artifact label. Use seed 42 for exploratory threshold choice, if any; seed 43 remains a prospective check. Compare to the previously inspected videos and disclose if the diagnostic rerun changes output hashes.

## Gate and cost

A signal does **not** justify selective fallback unless it enriches visually confirmed artifacts rather than normal gestures in the seed-43 check and does not miss obvious delayed seams. A threshold tuned on seed 43 or grayscale peak alone fails this gate. If it passes, separately benchmark a fallback's additional DiT passes, latency, VRAM, lip, identity, and visual motion on a held-out identity. If it fails, publish the negative result and stop before spending GPU on fallback. The diagnostic adds only elementwise reductions and JSON writing; measure its wall-time overhead cautiously because startup/cache variance can dominate. Thirty seconds cannot support an infinite-video claim.
