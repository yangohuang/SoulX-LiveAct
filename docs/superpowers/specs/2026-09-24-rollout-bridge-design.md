# Rollout-conditioned latent bridge on one RTX 4090

## Goal and boundary

Test whether a small trainable module can reduce LiveAct's block-start pose
jump using states from the model's own generated rollouts. Keep the 18B DiT
frozen and retain the three-step, 416×720 inference path. This is a controlled
algorithm experiment for long-video continuity, not a reproduction of Vidu
S2 Self-Replay Forcing: it has no DMD teacher, gradient-enabled causal DiT
replay, or cross-block gradients through the backbone.

The clean 4090 PR branch `feat/rtx4090-low-memory-kv` is separate and must
remain unchanged. Research proceeds on `codex/liveact-rollout-bridge` from the
existing boundary-diagnosis branch. No upstream PR without user review.

## Alternatives considered

1. Full DiT LoRA with on-policy replay would be closer to Vidu S2, but the
   current 18B FP8/offload runtime is inference-only and a single 24 GB 4090
   cannot retain full training activations across blocks. It also lacks a
   paired high-quality training set and a DMD teacher.
2. Another hand-tuned latent anchor is cheap, but earlier fixed, trend, and
   low-pass variants either trade away lip sync or move the pose jump later.
3. **Chosen pilot:** train a small residual bridge on captured generated
   latents. It makes the training-state source explicit, is cheap to fit, and
   can be rejected quickly on held-out identities and the known 15.58-second
   event.

## Data and model

Add a default-off latent-capture option that writes each completed block's
clean latent to CPU after denoising, without changing RNG or model operations.
Generate short rollouts for training identities 2 and 3, and validation
identity 4, using the same 4090 settings as prior tests. The existing
image-1 block-12 probe is the long-horizon held-out case.

Create pseudo-boundaries *inside* each generated block, where the original
sequence is relatively continuous. Corrupt the first two future latents by
mixing in non-neighbor latents from that same rollout. A compact convolutional
module receives two historical and three future latent states and predicts a
residual for the first two future states. Train against the uncorrupted states
with an identity term on uncorrupted examples. Initialize the final layer to
zero so the untrained module is a no-op. The model has no audio input; this
limits its expected ability to preserve speech motion and makes lip-sync
evaluation mandatory.

Use deterministic train/validation identity splits and save the exact model
configuration, seed, loss curves, and checkpoint. The pilot trains only this
module, never the DiT or VAE. Report GPU memory and training time.

## Evaluation gate

First apply the trained module **offline** to the saved image-1 block-12
latents, decode with the unchanged VAE, and inspect output frames around
373→374 and the next latent boundary at 376→377. Compare against unmodified
latents and fixed 0.25/0.45 anchor outputs. A lower single-frame spike is
insufficient if an equal or worse peak appears later. Also test an unseen
identity-4 short rollout. If these gates fail, stop before integration into
full generation, document the negative result, and leave default inference
untouched. If they pass, integrate opt-in and run full 30-second video,
SyncNet, and long-sequence motion checks before calling it a quality gain.

The interview claim is limited to the work actually verified: rollout-state
dataset construction, a frozen-backbone trainable correction pilot, and
measured generalization or failure. Do not call it Self-Replay Forcing.

## Observed decision

The pilot trained and passed the synthetic-corruption metric, but it also
altered uncorrupted validation latents and left the target real pose jump
visibly unresolved. The evaluation gate rejected inference integration.
See `docs/benchmarks/2026-09-24-liveact-rollout-bridge.md` for measurements.
