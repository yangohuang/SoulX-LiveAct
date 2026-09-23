# LiveAct 4090 FP8 Weight Offload Implementation Plan

> **For agentic workers:** Execute inline, task by task. Preserve the existing uncommitted 4090 low-memory work and generated media.

**Goal:** Reduce the 416×720 single-4090 14-frame block latency from the measured 629-second generation baseline while retaining a working 64 GB host-memory path.

**Architecture:** Quantize DiT block weights to FP8 once, discard BF16 sources, then keep the smaller FP8 weights in pageable CPU RAM. Keep the existing asynchronous two-GPU-slot prefetch. Allocate CPU KV cache only after BF16-to-FP8 conversion to reduce the initialization peak.

**Tech Stack:** PyTorch 2.8 CUDA, vLLM FP8 GEMM, Python unittest, ffmpeg/ffprobe.

---

### Task 1: Establish FP8-only behavior

- [x] Run the existing `--fp8_gemm --offload_cache --fp8_kv_cache --block_offload --t5_cpu --disable_compile` path with the 0.5-second 416×720 fixture.
- [x] Capture model initialization, per-step time, GPU memory, host memory, and whether the MP4 decodes.
- [x] FP8 path completed; the later two-block test exposed excessive shared pages from asynchronous GPU→CPU KV transfer. Reproduced with a 1 GiB tensor and wrote a failing GPU regression test before fixing both model variants.

### Task 2: Reduce initialization memory peak

- [x] Measure direct pageable-to-GPU and pageable-to-pinned-to-GPU transfers. On this 4090, direct transfer of 384 MiB took 26.6 ms; staging plus transfer took 35–40 ms. Do not add a slower staging path.
- [x] Add a failing test for optional audio-CFG KV cache allocation with the same shapes, dtypes, and independent dictionaries as the existing implementation.
- [x] Extract KV cache construction into a helper and invoke it after FP8 conversion and auxiliary-model setup.
- [x] Make GPU→CPU KV copies blocking to prevent unbounded pinned/shared pages; retain CPU→GPU asynchronous prefetch.
- [x] Run unit tests and `py_compile`.

### Task 3: Full 4090 validation

- [x] Run 416×720 with FP8 and deferred KV allocation on the same short input and seed as the baseline, then run two continuous blocks with safe GPU→CPU KV copies.
- [x] Verify MP4 resolution, frame count, audio/video decode, and a sample frame.
- [x] Compare block time and memory against the BF16 629-second baseline and FP8-only run.
- [x] Document the reproduced command, measured result, and remaining bottleneck in README and the benchmark report.
- [x] Run final tests and `git diff --check`; review all changed files while preserving preexisting edits.
