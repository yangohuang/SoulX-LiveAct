# RTX 4090: profile, startup cache, and denoising-step comparison

## Integrated-branch smoke test

The integrated branch was checked on the same RTX 4090 with three denoising
steps, the offline FP8 and prompt caches, `--resident_kv_steps 1`, and
`--stream_video_output`. For the 1.5-second audio fixture, cached startup to
ready was 6.996 s; the first and second blocks took 18.296 s and 19.192 s.
The output has 38 video frames at 24 fps and 1.500 s of audio. Both streams
passed complete FFmpeg decoding. This is a functional smoke test, not a new
same-run performance A/B or a quality assessment.

## What transfers from the RTX 5090 path

The [unofficial RTX 5090 optimization fork](https://github.com/5461gpt/SoulX-LiveAct-RTX5090)
reports a warm 256×416, 15 fps, three-step profile with a prequantized NVFP4
checkpoint, persistent compilation and conditioning caches, fixed-shape
warmup, VAE temporal decoding, and HLS output. Its [reproduction guide](https://github.com/5461gpt/SoulX-LiveAct-RTX5090/blob/main/docs/RTX5090_REPRODUCTION_GUIDE_ZH-TW.md)
specifies an SM120 GPU and reports roughly 2.1 s to a deliverable first
chunk. Those figures are not comparable to this 416×720, 24 fps, FP8 RTX 4090
run. NVFP4 kernels require Blackwell hardware and are not a portable 4090
optimization.

The transferable startup idea is to prepare weights and prompt embeddings
offline, which this branch implements in FP8. The transferable long-request
idea is bounded output buffering, implemented here by the optional block
writer. The fork's [prepared conditioning](https://github.com/5461gpt/SoulX-LiveAct-RTX5090/blob/main/docs/PREPARED_CONDITIONING_CACHE_BENCHMARK_ZH-TW.md)
and [VAE temporal cache](https://github.com/5461gpt/SoulX-LiveAct-RTX5090/blob/main/docs/VAE_TEMPORAL_CACHE_BENCHMARK_ZH-TW.md)
remain separate experiments: the latter changes the decoder's temporal
boundary behavior, and neither has been validated on this 4090 path. Our
profiler identifies weight and KV transfers as the first throughput targets.

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
generation on this RTX 4090.

## First-step KV residence: follow-up experiment

With the default three denoising steps unchanged, `--resident_kv_steps 1`
keeps only step 0's FP8 KV tensors on GPU. The other two step caches still use
the original CPU offload. This adds about 6.4 GiB persistent GPU KV storage
at `416*720` and requires FP8 KV, CPU cache offload, and audio CFG no greater
than 1.0.

| Run | Subsequent block median | First / last five subsequent blocks | Sampled GPU process peak |
| --- | ---: | --- | ---: |
| 5 s, CPU KV baseline | 21.341 s | 3 subsequent blocks | not sampled in this run |
| 5 s, first-step GPU KV | 18.891 s | 3 subsequent blocks | 17.13 GiB (1 s sampling) |
| 30 s, CPU KV baseline | 21.401 s | 21.495 / 21.417 s | 8.69 GiB |
| 30 s, first-step GPU KV | 18.954 s | 18.934 / 19.448 s | 16.12 GiB (3 s sampling) |

The 30-second resident run completed 23 blocks and produced 722 video frames,
30.083-second video, and 30.000-second audio. Complete FFmpeg decoding passed;
early, middle, and late frames showed no obvious identity collapse. Its
sampled minimum host `MemAvailable` was 34.45 GiB. A separate resident
profiling run measured the first DiT forward in the second block at 3.891 s,
versus 6.319 s on the CPU-offloaded baseline. The 40-layer KV H2D and D2H
profiler ranges disappeared; FP8 GEMM and block-weight copies remained about
0.97 and 1.26 seconds, respectively. Profiling overhead makes the isolated
forward times unsuitable as exact steady block predictions.

The resident 5-second video differs slightly at the pixel level from the CPU
KV output; sampled frames show no obvious degradation, but lip sync and
perceptual equivalence are not established. The last five blocks of the first
30-second resident run rose to a 19.448-second median, while a fresh 5-second
request returned to 18.86–18.96 seconds. A second 30-second run recorded
CUDA events for every DiT forward and profiled the first forward of block 20.
Excluding that profiled block, the last four blocks were 18.962, 18.973,
18.996, and 18.961 seconds. Early versus late CUDA-event medians were
3.849/3.853 seconds for step 0, 6.499/6.520 for step 1, and 6.478/6.480
for step 2. The late step-0 profiler measured 3.862 seconds and still showed
no KV transfer ranges. The one-run late slowdown did not reproduce, so its
cause remains unconfirmed. The overall median target below 19 seconds was
met, but an every-block 19-second bound has not been demonstrated.

A persistent process with resident KV completed two identical 1.5-second
requests in 44.508 and 36.900 seconds; their 38 decoded frames matched
pixel-for-pixel. An intervening prompt outside the pre-encoded catalog
returned `LIVEACT_ERROR` without terminating the worker.

## Pinned block weights: speed versus memory

One further 5-second A/B kept the same cached 3-step, resident-step-0 KV
configuration and changed only `--pin_block_memory`. The runs were sequential,
so minor OS and GPU variance remains possible. Raw logs and 2-second samples
are `/tmp/liveact-pin-ab-{baseline,pinned}.log` and corresponding
`-samples.csv` files.

| Setting | Steady block times | Median | Ready time | Minimum host `MemAvailable` | Sampled whole-GPU peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Default pageable block weights | 19.099, 18.990, 18.966 s | 18.990 s | 6.982 s | 37,199 MiB | 15,866 MiB |
| Pinned block weights | 16.858, 17.338, 16.585 s | 16.858 s | 19.761 s | 9,128 MiB | 21,195 MiB |

The pinned setting cut the three-block median by about 11%, but consumed about
27 GiB more host headroom, increased startup by about 13 seconds, and left
little margin for a longer output or another process. Both 117-frame videos
decoded fully. Their pixels differed, as did separate unpinned process runs;
the pinned run's 2.5-second frame had no obvious corruption on inspection.
That is insufficient to establish perceptual or lip-sync equivalence. Keep
pinning opt-in on this 64 GiB machine rather than combining it with long-video
generation by default.

## Stream decoded blocks to the encoder

The optional `--stream_video_output` writes each decoded block to imageio's
FFmpeg writer and discards the decoded tensor instead of retaining every block
for a final `torch.concat` and float NumPy conversion. A synthetic two-block
test compares the fully decoded frames with the existing Diffusers exporter
byte-for-byte. The writer is closed through a context manager even if block
generation raises.

A cached 5-second A/B with resident step-0 KV and the default three denoising
steps gave subsequent block medians of 18.977 seconds (whole-video export) and
19.017 seconds (streamed); both produced 117 frames, 4.875-second video and
audio, and passed full FFmpeg decoding. The runs were separate CUDA processes,
so their generated pixels are not identical; the identical synthetic test is
the direct encoder-parity check. A brief concurrent GPU task affected the
streamed run's early whole-machine RAM/GPU samples, so its sampled minimum
`MemAvailable` cannot be treated as a memory comparison. Raw files are
`/tmp/liveact-stream-ab-{baseline,streamed}.{log,mp4}` and the corresponding
`-samples.csv` files.

The streamed 30-second run completed 23 blocks in 461.4 seconds wall time.
Its subsequent block median was 19.025 seconds; the 722-frame, 30.084-second
H.264/AAC output decoded fully. Frames at 1, 15, and 29 seconds showed no
obvious identity or scene collapse. A same-version 30-second original-export
run then used the same fixture, caches, model settings, and 3-second process
sampler. Both videos have 722 frames at 24 fps and 30.000-second audio; both
fully decode. The MP4 container durations differ by 0.083 seconds because of
the last video packet timestamp, while both video track durations are exactly
30.083 seconds.

| 30-second run | Later-block median | Process `VmRSS` at ~61 s | At ~449 s | Sampled peak `VmRSS` | Sampled peak `RssAnon` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original whole-video export | 18.939 s | 36,154 MiB | 38,010 MiB | 40,607 MiB | 22,242 MiB |
| Stream each decoded block | 19.025 s | 36,265 MiB | 36,825 MiB | 36,831 MiB | 18,466 MiB |

The sampled peak process RSS fell by 3,776 MiB (3.69 GiB). From early to late
generation, the original path grew about 1,856 MiB, versus 560 MiB with
streaming; the extra original-path growth is consistent with retaining BF16
decoded frames. Its final concatenation/conversion caused the larger peak.
These are single sequential runs with 3-second samples, so brief peaks may be
missed and a repeat would refine the variance. The later-block medians differ
by 0.086 seconds, too little to claim a generation-speed change. Both runs had
zero sampled swap; the streamed run's `RssShmem` peaked at 18 MiB. Whole-machine
`MemAvailable` was perturbed by another workload during the streamed run, so
the table uses per-process RSS. Raw data are `/tmp/liveact-{baseline,stream}-30s.log`
and matching `-samples.csv` files.
