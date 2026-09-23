# RTX 4090: profile, startup cache, and denoising-step comparison

Measured locally on a 24 GB RTX 4090, 62 GiB host RAM, PyTorch 2.8.0+cu128.
The single-GPU command used `416*720`, 24 fps, seed 42, FP8 GEMM, FP8 CPU KV
cache, block offload, CPU T5, and disabled `torch.compile`. The comparison
fixture uses `examples/image/1.png`, prompt `一个人在说话`, and the first 5 seconds
of `examples/audio/1.wav`. The generated MP4 has 117 frames (4.875 seconds).
All paths and outputs in this report are local; no remote hardware was used.

## Where the steady block spends time

PyTorch profiler recorded CPU and CUDA activities for one DiT forward in the
second generation block. Its wall time was 6.319 seconds; profiling raised the
whole block to 23.966 seconds, compared with about 21.4 seconds without the
profiler. The categories are nested and may overlap, so their times must not
be summed or multiplied by three to obtain an exact block breakdown.

| Event in one forward | Calls | CUDA range / kernel time |
| --- | ---: | ---: |
| KV GPU-to-CPU transfer | 40 | 1.777 s (pageable D2H memcpy 1.725 s) |
| Block weight copies | 2,280 | 1.240 s |
| FP8 linear layers | 492 | 1.036 s (CUTLASS scaled GEMM 0.973 s) |
| KV CPU-to-GPU transfer | 40 | 0.567 s |
| SageAttention kernels | 120 | 0.292 s |

The active path used SageAttention; the instrumented SDPA fallback was not
called. Transfer cost is a better next target than switching attention
kernels. The forward was the first denoising step, so later audio-conditioned
steps may have different attribution.

## Startup and request reuse

The offline FP8 DiT cache occupies 18 GB and stores quantized linear weights
plus non-linear tensors. It loads into a meta-device model, avoiding BF16 DiT
shard allocation and runtime quantization. A separate small prompt cache skips
T5 loading for an exact pre-encoded prompt catalog. Source size and mtime are
checked before cache reuse. The following runs were sequential on the same
machine; Linux file-page caching changed between runs, so they are not
independent cold-disk measurements.

| Run | Startup to ready | Relevant phase | Request / generation |
| --- | ---: | --- | --- |
| Build FP8 cache, T5 uncached | 281.225 s | T5 139.428 s; FP8 block quantization 94.476 s; cache write 5.624 s | 1.5 s audio: 20.792 s first block, 21.465 s second block |
| Load FP8 cache, build prompt cache, keep process resident | 175.713 s | T5 140.488 s | 1st request 55.382 s; identical 2nd request 41.982 s |
| Load both caches, 2-step 5 s run | 37.516 s | T5 cache lookup 0.107 s; DiT cache mapping 0.350 s | 4 blocks below |
| Load both caches again, 3-step 5 s run | 6.790 s | T5 lookup 0.091 s; DiT mapping 0.288 s | 4 blocks below |

The DiT mapping time alone is misleading: mapped pages are faulted in during
later setup and inference. The first fully cached launch took 37.5 seconds to
ready; a later launch with warmer OS file pages took 6.8 seconds. The build
run reached 54.4 GiB maximum process RSS on the 62 GiB host. In the persistent
process, the second 1.5-second request reused models and prompt embeddings;
after resetting KV caches, its decoded 38 frames matched the first request
pixel-for-pixel.

## Two versus three denoising steps

The two-step schedule keeps the first and final denoising conditions:
`[1000, 833.33333333, 0]` with audio skipped only in the first forward.
The default three-step schedule remains `[1000, 937.5, 833.33333333, 0]`.
Both 5-second runs used the same code, offline caches, media, resolution, and
seed. Their startup times differ because the OS file cache warmed; compare
the generation blocks, not whole-process times.

| Steps | Block 0 | Subsequent blocks | Subsequent median | Generated frames/s |
| --- | ---: | --- | ---: | ---: |
| 3 (default) | 20.399 s | 21.420, 21.341, 21.311 s | 21.341 s | ~1.50 |
| 2 (experimental) | 14.795 s | 14.925, 14.821, 14.753 s | 14.821 s | ~2.16 |

Two steps reduced steady block time by 30.6%. Both videos have 117 frames,
4.875-second video duration, audio, and passed complete FFmpeg decoding.
Samples at 0.5, 2.5, and 4.0 seconds show no obvious identity collapse, but
the mean absolute decoded-frame change between adjacent frames rose from
2.52 to 2.94 pixel values on a 0–255 scale. That diagnostic does not measure
perceptual quality or lip sync. Motion differs visibly, and a proper listening
and lip-sync review remains necessary before changing the default. The
side-by-side local artifact has three steps on the left and two on the right.

Even the faster setting is about 11 times slower than 24 fps real-time
generation on this RTX 4090. The next measurable target is to reduce the
default three-step steady block below 19 seconds while staying under 18 GB
GPU process memory, first by testing one GPU-resident denoising-step KV cache
against the current CPU-offloaded baseline. This is an experiment, not an
achieved result.
