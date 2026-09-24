# Diagnose and reduce a LiveAct chunk-start pose jump

## Objective

Explain the 15.58-second head/hand pose discontinuity in the 30-second RTX
4090 fixture and find a quality-preserving local mitigation if one exists.
Keep the memory-fix PR branch untouched. Use original three-step inference,
416×720 output, seed 42, and cached FP8 weights for controlled comparisons.

## Observations

The initial block yields 21 video frames; later blocks add 32. Frame 373 is
the first output frame of a later block, and the largest appearance change is
between frames 373 and 374, inside the first new temporal latent. Changing
only VAE history from three to five latents reduced the short-run boundary
proxy by 4.2% but left large spikes. BF16 KV history did not improve the
short-run proxy. A fixed clean-latent anchor lowers many seams but does not
eliminate this pose event and stronger anchoring worsens SyncNet.

## Diagnostic method

Add an opt-in probe at one chosen generated block. Save the previous block's
final two clean latents, the new block's first two predicted clean latents
after each denoising step, the final two generated latents, and metadata on
step, block, and frame mapping. The probe must not draw random numbers or
alter the model's result. Compare a probe-enabled output with a same-code,
same-seed baseline at the decoded-frame level and report any nondeterminism
rather than assuming exact pixel identity.

Compare temporal distances within the previous block, across the boundary,
and inside the new block at each step. Relate those distances to decoded
frame-change peaks and inspect whether the first new latent is already
discontinuous before VAE decoding. If latent evidence is inconclusive,
avoid attributing the jump to one subsystem.

## Diagnosis result and selected mitigation

The reproduced 30-second run has a generated latent boundary RMSE of 0.434,
versus 0.156 in the previous last pair and 0.307 in the new first pair. The
largest output change remains at frame 373→374. Re-decoding the exact same
final latent with three, four, and five prior latents yields first→second
new-frame RGB changes of 15.581, 15.352, and 15.277 on a 0–255 scale.
The VAE history length does not explain the event. The generated first new
latent carries a different pose trajectory.

The selected local mitigation filters the existing anchor correction in the
spatial latent plane before applying it. A 5×5 low-pass should constrain
coarse pose and hand position while leaving more high-frequency latent detail
for speech articulation. Compare it with no anchor and the existing full
anchor at the same strength. This is only an inference-time quality/speed
trade-off; a trained on-policy method remains a separate research direction.

## Fix decision gate

Only test a mitigation that targets the stage implicated by the probe.
Require a visible improvement at the 373→374 event and at least one other
identity, with no clear lip-sync regression. Keep all changes opt-in until
longer examples support them. If no method meets the gate, document the
failure and the likely need for training-time autoregressive correction.
