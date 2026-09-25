# Velocity-aware boundary anchor: small inference ablation

## Motivation and hypothesis

The fixed latent anchor pulls the first two predicted clean latents of every new chunk toward the *last* latent of the previous chunk. In the 30-second pilot and the second-seed pairs available so far, this reduces grayscale frame-change peaks but also reduces local motion, hurts the SyncNet proxy and smears moving hands. A plausible mechanism is that a static target conflicts with a continuing gesture. This remains a hypothesis, not an established model failure cause.

The candidate uses a first-order trajectory target from the last two generated latents: `velocity = last - penultimate`; target new latent 0 is `last + velocity`, and target new latent 1 is `last + 2*velocity`. As with the existing anchor, the first target is blended with requested strength and the second with one third of it. Other latents remain unchanged. Zero strength keeps the original path. The option is explicit and opt-in; the existing fixed anchor and default behavior must stay byte-for-byte equivalent in code paths and unit tests. It adds no DiT forward pass, only latent arithmetic, and should not claim zero latency without measurement.

## Scope and acceptance

Add a small pure function or mode to `temporal_continuity.py`, a CLI choice in `generate.py`, unit tests for a constant-velocity toy trajectory, zero-strength identity, input validation and preservation of the original fixed mode. The generated videos are compared on image 2, seed 43, 30 seconds, with the already completed baseline and fixed 0.45 anchor, all other inputs and runtime settings unchanged. Record 722-frame validity, the same 22 boundary windows, delayed peaks and motion sum, 31-point face proxy, 25-FPS SyncNet crop comparability, visual hand sequence at the baseline's worst window 500, and generation wall time. If hand smearing remains or lip synchronization worsens relative to baseline, reject the candidate despite a smaller grayscale peak. If the candidate passes this development case, run image 3 as a held-out identity; otherwise stop the branch of experiments without threshold tuning.

This is a first-order inference heuristic, **not** Self-Replay Forcing, model retraining, a physics prior, or a Vivix-W1 implementation. A negative result still tests why naive latent continuity constraints can conflict with action fidelity.
