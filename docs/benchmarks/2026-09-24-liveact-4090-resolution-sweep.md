# RTX 4090 spatial-resolution throughput sweep

This is a local throughput probe, not a quality comparison. All runs used one
24 GB RTX 4090, `examples/image/1.png`, the same 5-second excerpt of
`examples/audio/1.wav`, seed 42, 24 fps, the cached FP8 DiT and prompt,
FP8 GEMM, FP8 KV, CPU offload for the remaining steps, first-step resident KV,
block weight offload, CPU T5, and no `torch.compile`. The 416×720 three-step
reference is the earlier matching five-second resident-KV run. Each new run
produced 117 H.264 frames with AAC audio; complete FFmpeg decoding passed.

The model uses spatial VAE stride 8 and DiT patch size 2×2, so one latent
frame has `(height/16) × (width/16)` DiT tokens. Every later generation block
samples eight latent frames and adds 32 decoded video frames. A 24-fps stream
therefore has a **1.333-second** compute budget per block, before accounting
for any desired first-response latency.

| Output | Steps | Tokens per latent frame | Later-block times (s) | Median (s) | Generated fps | Gap to 24 fps |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 416×720 | 3 | 1,170 | earlier five-second run | 18.891 | 1.69 | 14.2× |
| 288×512 | 3 | 576 | 11.174, 11.059, 11.017 | 11.059 | 2.89 | 8.3× |
| 224×384 | 3 | 336 | 7.296, 7.498, 7.109 | 7.296 | 4.39 | 5.5× |
| 224×384 | 2 | 336 | 5.081, 4.928, 4.787 | 4.928 | 6.49 | 3.7× |

At 288×512, DiT tokens fall to 49.2% of the 416×720 count, but block time
only falls to 58.5%. At 224×384, tokens fall to 28.7% and three-step time
to 38.6%. The fixed model size and transfers prevent a proportional speedup.
Output-only downscaling after VAE decode does not reduce DiT work. Lowering
the generated latent resolution does, as these runs show, but the quality,
lip sync, identity, and motion continuity of the smaller outputs have not
been assessed. The two-step schedule was already known to change motion at
416×720 and is experimental, not a quality-preserving default.

## Lower-resolution profiler

The same PyTorch profiler harness recorded the first DiT forward of the
second block at 224×384, three steps, resident step-0 KV. Its wall time was
1.988 s; recorded CUDA time for the forward was about 1.987 s. Within that
forward, 2,280 block-weight copies accounted for 1.196 s of CUDA ranges,
492 FP8 linears for 0.361 s, and 120 SageAttention kernels for 0.030 s.
The categories are nested and CUDA copies can overlap compute, so they are
not additive wall-time components or a strict lower bound. They do show
that weight staging is the dominant recorded operation at this size; simply
switching attention kernels cannot close the 3.7–5.5× real-time gap.

The earlier 416×720 CPU-KV profile of the same first forward recorded
1.777 s KV D2H, 0.567 s KV H2D, 1.240 s weight copies, 1.036 s FP8 linears,
and 0.292 s SageAttention kernels. Keeping step-0 KV on GPU removed its KV
transfer ranges and lowered that forward from 6.319 to 3.891 s under the
profiler. The later audio-conditioned steps still use CPU KV. Separate CUDA
events measured about 3.85, 6.5, and 6.48 s for steps 0, 1, and 2 at
416×720. These event measurements and block timings came from different
runs, so their difference is only an approximate allowance for VAE decode,
sampling operations, CPU scheduling, and other work.

The next engineering experiment is to quantify weight residency at low
resolution and the remaining two steps' KV transfers without changing
model outputs. Reaching 24-fps generated throughput or subsecond first
response on one 4090 likely requires changing the model/runtime jointly:
fewer effective model passes, smaller or distilled backbone, sparse-frame
generation with interpolation, or a causal smaller-chunk decoder. None of
those options is established by this sweep, and each needs quality and
audio-alignment checks.
