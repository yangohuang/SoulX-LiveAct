# Pinned block weight A/B plan

- [x] Verify the existing `--pin_block_memory` path, fixed inputs and cache sources; inspect GPU/host headroom.
- [x] Run fresh pageable and pinned 224×384 three-step arms with resource sampling and the same input. Verify decoded media and per-block timings.
- [x] Compare median later-block time, host/GPU memory, and decoded frame equivalence. The low-resolution speed/headroom gate passed; a same-arm baseline repeat quantified unexpected decoded-frame variation.
- [x] Repeat at 416×720 under the same safety, output and timing checks. The target-resolution 15% speed gate failed (13.0% gain), so no long-run quality promotion.
- [x] Save report, machine-readable metrics and visual evidence; update roadmap/interview evidence and run appropriate checks. Sync only the personal fork.
