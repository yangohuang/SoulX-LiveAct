# Motion-trend boundary conditioning on one RTX 4090

## Objective

Study autoregressive chunk-boundary motion continuity without increasing the
three-step DiT cost. The result should be a measured, opt-in inference experiment,
useful for discussing temporal error accumulation and quality/speed trade-offs in
the Vivix video-generation interview. It is separate from the upstream 4090
memory-fix PR candidate.

## Evidence and scope

LiveAct's Neighbor Forcing is a training method that propagates same-step
neighbor states. Vidu S2's Self-Replay Forcing is an on-policy training method
requiring rollout, re-noising, differentiable replay, and distillation. Vivix-W1
discusses long-horizon error accumulation but does not disclose a forcing
implementation. This experiment is inspired by generated-history conditioning;
it does not claim to implement either paper's training method.

The current local fixed anchor blends the next chunk's first two predicted
clean latents toward the previous chunk's final latent at every denoising step.
At strength 0.25, one 30-second sample reduced the mean first-eight-frame
appearance-change proxy by 17.8%, while lip-sync evidence was mixed. A stronger
anchor harmed the measured boundary lip-sync score. The fixed anchor remains
off by default.

## Design

Add an optional trend coefficient to the existing anchor. With previous clean
latent frames `p[-2]` and `p[-1]`, estimate `delta = p[-1] - p[-2]`. The first
target is `p[-1] + trend * delta`; the second is `p[-1] + 2 * trend * delta`.
Blend the two new predicted clean latents toward these targets with the
existing weights `strength` and `strength / 3`. `trend=0` reproduces the
existing fixed anchor exactly. Require at least two previous latents when
`trend>0`. Keep the default `strength=trend=0` bitwise on the original path.

The hypothesis is that a short linear latent trajectory may prevent the
freeze-and-jump behavior of copying the final frame. It can also overshoot,
misrepresent articulated motion, and disturb speech. Therefore this is an
ablation, not a new default or a quality claim.

## Validation

Use identical image, audio, seed, resolution, steps, FP8 settings, and cache
settings across no anchor, fixed anchor, and trend anchor. Run short pilots on
both supplied example identities, then one 30-second run if the pilots are
reasonable. Report chunk-boundary and non-boundary frame-change proxies,
worst seams, generated-frame throughput, full MP4 decode, visual motion notes,
and the existing SyncNet pipeline when available. Compare to existing
30-second fixed-anchor measurements only with a matching setup and explicitly
mark any cross-run environmental differences. A lower pixel-change proxy alone
does not prove better motion or lip sync.

Success means the trend option improves visible boundary motion on both pilot
identities without material lip-sync degradation and without added DiT forwards.
If it fails, keep the negative result as the finding and leave the original
default unchanged.
