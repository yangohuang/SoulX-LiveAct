# Audio lookahead and phase audit plan

- [x] Write failing tests for block-0 and later-block audio index/decoded-frame ranges, 30-second period phases and 32-second alignment.
- [x] Implement a read-only analyzer driven by existing LiveAct constants and save 30/90 and 32/96 JSON with source-code references and measured 4090 block timing.
- [x] Check the analyzer's 10/20 endpoint-clamp prediction against archived frame-audit hashes; inspect 30/90 and 32/96 boundary-spanning blocks.
- [x] Document implementation-level audio availability, block scheduling cost, phase confounder and a conditional next experiment without claiming measured interactive latency or long-history degradation.
- [x] Update Vivix evidence card and memory, verify tests/report hashes, then commit and sync only the personal fork and verify a clean worktree.
