# RTX 4090 block-boundary pose jump: diagnostic record

## Reproduction and stage localization

Fixture: `examples/image/1.png`, the 30-second local audio fixture, 416×720,
24 fps, seed 42, three denoising steps, cached FP8 DiT, and resident KV for
step 1. The generated video contains 722 source frames. Use `ffmpeg -vsync 0`
for frame-difference measurements; this FFmpeg build otherwise duplicates
two final frames when decoding the unanchored video to rawvideo. The visible jump
at frames 373→374 (15.54→15.58 s) recurs in the instrumented run. The two
videos are not pixel identical; the probe adds tensor copies and file writes
but no RNG call or model change. A paired 1.5-second run with and without
the probe produced 38 frames each, 2.81/255 mean RGB difference and 32.76 dB
PSNR. Differences start in block 0, before the selected block-1 probe runs,
so this check shows cross-run variation rather than a probe-specific effect.
It does not establish bitwise determinism. The same-run latent measurements
support stage localization, not a deterministic-output claim.

`--boundary_probe_block 12 --boundary_probe_dir <path>` records the prior
block's last clean latents and the new block's first denoised latents. The
transition RMS error is:

| Stage | Previous adjacent | Cross-block | New adjacent | Cross / adjacent mean |
| --- | ---: | ---: | ---: | ---: |
| First predicted clean latent, t=1000 | 0.156 | 0.358 | 0.209 | 1.97× |
| Second, t=937.5 | 0.156 | 0.440 | 0.306 | 1.91× |
| Final, t=0 | 0.156 | 0.434 | 0.307 | 1.88× |

The first clean-latent prediction already contains an unusually large
cross-block change. The final cross-block error exceeds every within-block
adjacent error in the five previous and eight new latents. Both coarse and
fine spatial components rise: 5×5 low-pass RMS is 0.302 across the boundary
versus 0.108 in the previous pair; the corresponding residual RMS values
are 0.271 and 0.103. This is an observed latent mismatch, not proof of one
specific causal mechanism inside the model.

Re-decoding the *same* final latents with three, four, and five historical
latents gives first→second new-frame RGB mean absolute changes of 15.581,
15.352, and 15.277 (0–255 scale). The pose change remains. More VAE history
does not address this example.

## Low-pass latent anchor pilots

The candidate smooths only the spatial low-frequency part of the existing
clean-latent anchor correction. It is opt-in, with no extra DiT forward.
The measurement below is the mean grayscale absolute difference in the first
eight transitions of each generated block, computed at 180×104. Lower can
mean less abrupt motion, but can also mean over-smoothing, so it is only a
screening metric. Each pilot is about five seconds, 117 decoded frames,
three measured boundaries.

| Identity | No anchor | Full anchor 0.25 | Full anchor 0.45 | Low-pass 5×5, 0.45 | Low-pass 5×5, 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image 1 | 2.661 | 2.343 | 2.040 | 2.376 | 2.146 |
| Image 2 | 3.434 | 2.790 | — | 3.095 | 2.908 |

Full anchor 0.45 gives the smallest image-1 proxy, but previous SyncNet
testing found worse lip alignment. For the low-pass 0.45 candidate, SyncNet
minimum distance/confidence was 7.540/6.301 for image 1 and 7.895/6.708
for image 2, versus 7.576/6.166 and 7.977/6.485 with full anchor 0.25.
At strength 0.80 the image-1 proxy improves over full anchor 0.25, with
SyncNet 7.566/6.423; the image-2 proxy remains worse and its SyncNet is
8.111/6.454, also slightly worse than full anchor 0.25. The first
low-pass setting traded worse measured continuity for slightly better
SyncNet scores on both short clips. Do not promote this option into a PR or
default without a replicated continuity and visible-quality gain.

## Targeted 30-second check

The 0.80 low-pass video completed in 8:14.79 wall time on the RTX 4090 and
has 722 source frames at 24 fps. FFmpeg decodes it without error. The table
uses 180×104 grayscale frames with `-vsync 0` for all inputs. Different runs
with the same seed are not guaranteed pixel-identical.

| Setting | First 8 mean | 373→374 | 376→377 | Largest adjacent change |
| --- | ---: | ---: | ---: | ---: |
| Probe baseline | 2.237 | 14.630 | 1.071 | 14.630 |
| Full anchor 0.25 | 1.821 | 10.368 | 1.565 | 10.368 |
| Full anchor 0.45 | **1.586** | 5.903 | 2.740 | **6.002** |
| Low-pass 0.80 | 1.815 | **4.147** | 7.339 | 7.339 |

The low-pass correction suppresses the original 373→374 transition but
delays the hand movement to 376→377. The contact sheet at
`/home/yg/yg/code/docs/liveact-4090-local/liveact-lowpass-target-frame371-378.png`
shows that the subject keeps her hand low for three more frames, then raises
it abruptly. The largest adjacent-frame change is worse than with the full
0.45 anchor, and the second identity also failed the short-clip gate. The
candidate does **not** solve the temporal continuity problem. Keep it as a
default-off research option only; do not recommend it for production or
claim a quality gain in an upstream PR.

## Interpretation

Forcing-style training would address a different level: the model could learn
to predict under its own rolled-out history, reducing long-horizon state
drift. The current anchor is an inference-only correction and should not be
described as Self-Replay Forcing or a training algorithm. The 4090 evidence
isolates the failure and gives a measurable target for later work; the tested
inference-only corrections do not yet establish a fix for the 15.58-second
pose jump. The next research step should train and evaluate under generated
history, with an explicit long-video continuity and lip-sync gate, rather
than tune an anchor solely against a pixel-difference proxy.
