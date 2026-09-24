# Frozen-backbone rollout-conditioned bridge pilot on RTX 4090

## Purpose and method

The earlier block-12 probe found a large state mismatch before VAE decoding:
cross-block clean-latent RMSE 0.434 versus 0.156/0.307 within the adjacent
latent pairs. Fixed and low-pass inference anchors did not reliably remove
the resulting hand/pose jump. This pilot asks whether a small trained module
can infer a better transition from LiveAct's own generated trajectories.

The 18B DiT and VAE remain frozen. `--rollout_latent_dir` captures every
generated block's final latent, default-off. A 101,536-parameter 2D
convolutional bridge sees the previous two and current first three latent
states and predicts residuals for the current first two. Training constructs
pseudo-boundaries **within** generated blocks, corrupts the first two future
states by mixing non-neighbor states, and reconstructs the original states.
Twenty percent of examples are uncorrupted. The last layer starts at zero,
so the untrained bridge is a no-op. This is generated-state exposure with a
supervised synthetic-corruption loss, **not** Vidu S2 Self-Replay Forcing:
there is no re-noised DiT causal replay, DMD supervision, or gradient through
the generator.

## Reproducibility

Generate eight-second, 416×720, 24-fps, three-step baseline rollouts with
FP8 GEMM/KV, CPU block offload, cached FP8 DiT/T5, and one KV step resident.
Images/audio 2 and 3 are training identities; image/audio 4 is a held-out
validation identity. Each rollout has seven blocks and 54 latent frames
(16×time×90×52), and each video has 194 frames. The capture paths are
`/tmp/liveact-rollout-data/image{2,3,4}`. Generation wall times were
3:09.76, 2:31.25, and 2:31.29, respectively. All three completed on the
single RTX 4090.

For each `i` in `2 3 4`, trim the paired WAV and create a one-item JSON
request with the paired image and a distinct output path:

```bash
ffmpeg -y -v error -i examples/audio/${i}.wav -t 8 -acodec pcm_s16le \
  /tmp/liveact-rollout-audio${i}-8s.wav
```

The JSON at `/tmp/liveact-rollout-image${i}-8s.json` contains
`[{"prompt":"一个人在说话","cond_image":"examples/image/2.png","cond_audio":"/tmp/liveact-rollout-audio2-8s.wav","output_path":"/tmp/liveact-rollout-image2-8s.mp4"}]`
for `i=2`, with each `2` replaced by the chosen identity. Then run:

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \
  /home/yg/miniforge3/envs/liveact/bin/python generate.py \
  --size '416*720' --ckpt_dir checkpoints/LiveAct \
  --wav2vec_dir checkpoints/chinese-wav2vec2-base --fps 24 \
  --input_json /tmp/liveact-rollout-image${i}-8s.json \
  --fp8_gemm --fp8_kv_cache --offload_cache --block_offload --t5_cpu \
  --disable_compile --fp8_cache_dir /tmp/liveact-fp8-cache \
  --prompt_cache_dir /tmp/liveact-prompt-cache --denoising_steps 3 \
  --resident_kv_steps 1 --seed 42 \
  --rollout_latent_dir /tmp/liveact-rollout-data/image${i}
```

Training command:

```bash
/home/yg/miniforge3/envs/liveact/bin/python train_rollout_bridge.py \
  --train_dir /tmp/liveact-rollout-data/image2 /tmp/liveact-rollout-data/image3 \
  --validation_dir /tmp/liveact-rollout-data/image4 \
  --checkpoint /tmp/liveact-rollout-bridge/model.pt \
  --steps 300 --batch_size 4 --hidden 64 --lr 0.001 \
  --eval_every 25 --validation_samples 32
```

Seed 42. Training took 1.19 seconds after data collection and peak CUDA
allocated memory was 54,538,240 bytes for the bridge process. The checkpoint
is preserved with this research branch at
`docs/benchmarks/artifacts/liveact-rollout-bridge-model.pt`,
SHA-256 `94287e7bc9a913db2d5180c6599eea88002104451f56081b6afb91a7ce5ecf7a`.

## Held-out synthetic corruption

On 32 fixed validation samples from image 4, mean latent L1 error fell from
0.19645 without correction to 0.18253 with the trained bridge (7.1%). On
uncorrupted image-4 windows, however, the bridge introduced 0.03356 mean L1
change. That drift exceeds the 0.01392 absolute gain on corrupted examples.
The model has learned some denoising of the artificial perturbation, but its
correction is not selective enough for clean motion.

## Actual generated boundaries

The image-1 block-12 probe, around output frame 373, was never in training
or validation. Re-decoding its captured latents with the same VAE gives
first-new-frame→second-new-frame RGB mean absolute change 15.581 on the
0–255 scale. The bridge lowers this to 13.536 (13.1%), but the hand and head
still change pose in one frame. The largest transition in the first twelve
new-frame changes remains 13.536; no later rebound was observed in that
short window. A directly openable comparison is
`docs/benchmarks/artifacts/liveact-rollout-bridge-block12.png`.

Three real boundaries of held-out image 4 have these first-twelve-frame
maximum RGB changes:

| Boundary | Unmodified | Bridge |
| --- | ---: | ---: |
| 1 | 7.822 | 7.952 |
| 2 | 9.745 | 9.637 |
| 3 | 8.217 | 8.037 |

The changes are small and mixed. In the image-4 contact sheets, expression
and hand movement look nearly the same. The learned module does not pass the
predefined real-boundary gate: the known severe pose jump remains visible,
and the uncorrupted validation drift is material. Therefore it was **not**
integrated into `generate.py`, and no full 30-second or SyncNet claim is made
for the bridge.

## Research conclusion

The generated-state dataset and frozen-backbone training loop are feasible
on a 4090. Improvement on synthetic boundary corruption does not transfer
strongly to a real model-generated pose discontinuity. A likely mismatch is
that within-block donor mixing does not reproduce the distribution or causal
origin of real cross-block errors; this is an inference from the results,
not a proven root cause. A serious forcing experiment needs training targets
and optimization tied to the generator's actual autoregressive states, with
lip-sync and long-video motion checks. Vidu S2 uses detached autoregressive
rollout, re-noising, and differentiable causal replay with DMD supervision;
this pilot does not implement those components.

Validation of the research code: `python -m unittest discover -s tests -v`
completed with 42 tests passing; `python -m py_compile` and `git diff
--check` completed without errors. The clean 4090 PR-ready branch was not
modified, and this experiment is not suitable for an upstream quality PR.
