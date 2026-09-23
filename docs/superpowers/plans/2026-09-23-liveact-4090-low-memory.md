# LiveAct RTX 4090 Low-Memory Execution Plan

**Goal:** Generate and verify a real LiveAct video on one RTX 4090 with 64 GB host RAM, then measure whether 416x720 is viable.

**Constraints:** Preserve the existing SDPA fallback, CPU T5 pre-encoding, and FP8 experiments. Avoid loading all KV cache on the GPU and avoid pinning all 40 DiT blocks in host memory. Keep the normal high-throughput behavior available behind explicit flags.

## Task 1: Add testable runtime controls

- Add CLI controls to disable first-run `torch.compile` and to opt into pinning CPU block weights.
- Add focused parser/compile helper tests.
- Run the tests and confirm they fail before implementation and pass afterward.

## Task 2: Remove the block-offload host-memory spike

- Extend both single-GPU and sequence-parallel block managers with a `pin_cpu_memory` option.
- Preserve the existing manager API default for external callers; make the 4090 launcher use unpinned CPU blocks by default.
- Add a CPU-only unit test proving that the helper skips or invokes pinning as requested.

## Task 3: Run a minimal real inference

- Create a temporary sub-second audio/input fixture from the supplied example.
- Run eager inference at 384x384 with FP8 CPU KV cache and unpinned block offload.
- Monitor GPU and host RAM, inspect the generated MP4 with `ffprobe`, and record elapsed time.

## Task 4: Measure the requested resolution

- Repeat the short inference at 416x720.
- If it completes, verify the MP4 and record peak resources and runtime.
- If it cannot complete safely, capture the exact failing stage and measured resource limit.

## Task 5: Document and verify

- Add a reproducible RTX 4090 command and the measured limitations to the README.
- Run unit tests, syntax/import checks, and media validation.
- Review the final diff without altering unrelated existing work.
