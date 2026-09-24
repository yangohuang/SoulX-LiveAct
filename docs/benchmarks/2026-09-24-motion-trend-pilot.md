# RTX 4090 motion-trend boundary pilot — 2026-09-24

## Question and provenance

Can a short extrapolation of generated latent motion reduce LiveAct's visible
chunk-boundary jumps more safely than the existing fixed latent anchor? This
is an inference-time experiment inspired by the general problem of
autoregressive error accumulation. It is **not** [Vidu S2 Self-Replay
Forcing](https://arxiv.org/html/2609.11638): that method trains on student
rollouts with re-noising and differentiable replay. [Vivix-W1](https://vivix.ai/tech-report-vivix-w1)
discusses long-horizon continuity but does not publish a forcing formula to
reproduce. [SoulX-LiveAct](https://arxiv.org/html/2603.11746) itself uses
same-step Neighbor Forcing and ConvKV Memory.

## Method

The existing optional fixed anchor blends the first two predicted clean
latents of a new chunk toward the final clean latent of the previous chunk.
The new `--motion_anchor_trend` option changes the target to a short linear
extrapolation of the previous final two latents. Both options keep the same
three DiT forwards per chunk and the original path is unchanged when anchor
strength is zero. `trend=0` is exactly the fixed anchor.

Two 5-second fixtures used `examples/image/{1,2}.png`, their matching audio
excerpts from `examples/audio/{1,2}.wav`, the prompt `一个人在说话`, seed 42,
416×720, 24 fps, the original three-step schedule, FP8 GEMM and CPU KV,
block offload, and one GPU-resident FP8 KV step. The comparison uses existing
no-anchor/fixed-anchor outputs and two new trend runs at strength 0.25,
trend 0.5. FP8 and prompt caches were reused. The first new run had colder
OS file pages, so its full-process time is not a speed comparison; later
block times were about 19.1 seconds on both trend runs.

For image 1, the trend run used:

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 python generate.py \
  --size '416*720' --ckpt_dir checkpoints/LiveAct \
  --wav2vec_dir checkpoints/chinese-wav2vec2-base --fps 24 \
  --input_json /tmp/liveact-trend-025-05-5s.json \
  --fp8_gemm --fp8_kv_cache --offload_cache --block_offload --t5_cpu \
  --disable_compile --dura_print --denoising_steps 3 --resident_kv_steps 1 \
  --fp8_cache_dir /tmp/liveact-fp8-cache \
  --prompt_cache_dir /tmp/liveact-prompt-cache \
  --motion_anchor_strength 0.25 --motion_anchor_trend 0.5
```

The JSON contains the image, prompt, and 5-second audio excerpt described
above, plus `output_path`. Image 2 changes only the two media paths and the
output path. The media excerpts and cache directories are local test assets;
the public repository supplies the original example media.

The new `continuity_metrics.py` decodes each MP4 to 180×104 grayscale and
computes mean absolute adjacent-frame change. A later chunk begins at
transition indices 20, 52, 84, etc. The first-window score averages eight
transitions from each boundary; the later-window score averages the remaining
24. This proxy detects abrupt appearance changes but does not measure
physical motion correctness, identity, or lip sync. Scores below were all
recomputed with the same script; they differ from older figures produced by
a different resize/grayscale pipeline.

| Fixture and setting | First 8 transitions | Later 24 | Mean seam | Worst seam |
| --- | ---: | ---: | ---: | ---: |
| Image 1, fixed 0.25 | 2.343 | 1.783 | 1.899 | 2.168 |
| Image 1, trend 0.5 + anchor 0.25 | 2.283 | 1.814 | 1.912 | 2.167 |
| Image 2, no anchor | 3.434 | 3.464 | 2.923 | 3.390 |
| Image 2, fixed 0.25 | 2.790 | 3.459 | 2.121 | 2.317 |
| Image 2, trend 0.5 + anchor 0.25 | 2.797 | 3.474 | 2.433 | 2.654 |

The trend reduced image 1's first-window proxy by only 2.5% relative to the
fixed anchor and increased its seam mean slightly. On image 2 it did not
reduce the first-window proxy and increased the mean seam by 14.7%. Both new
MP4s contain 117 H.264 frames, AAC audio, and passed full FFmpeg decoding.

The same 25-fps face-crop SyncNet pipeline used for the earlier anchors found
offset -2 converted frames for both trend runs. For image 1, the fixed/trend
minimum distances were 7.576/7.625 (lower is better), while confidences were
6.166/6.286 (higher is better). For image 2, the distances were 7.977/7.973
and confidences 6.485/6.381. The metrics move in opposite directions and do
not establish a lip-sync improvement. Five-second clips are too short for a
long-horizon alignment claim.

## Decision

The trend hypothesis has not beaten the fixed anchor on both identities, so
we stopped before a 30-second run. The option remains default-off on this
isolated research branch and should not be merged into the 4090 memory PR.
For a stronger next experiment, measure motion with tracked landmarks or
optical flow and test a spatially selective or learned boundary condition;
pixel-change alone rewards frozen motion. Training-time replay forcing would
need a separate dataset, quality target, and memory budget on the 4090.

Raw runs and videos are local:

- `/tmp/liveact-trend-025-05-5s.{json,log,mp4}`
- `/tmp/liveact-second-trend-025-05-5s.{json,log,mp4}`
- `/home/yg/yg/code/docs/liveact-4090-local/liveact-fixed-left-trend-right-5s.mp4`
- `/home/yg/yg/code/docs/liveact-4090-local/liveact-second-fixed-left-trend-right-5s.mp4`
