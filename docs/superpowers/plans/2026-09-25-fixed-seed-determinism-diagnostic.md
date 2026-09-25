# Fixed-seed determinism diagnostic plan

- [x] Freeze two-block input, current flags and model source fingerprints; implement read-only tensor fingerprint hooks.
- [x] Run two identical 4090 processes, compare fingerprints in causal order, and confirm the short media reproduces the larger same-arm discrepancy.
- [x] Isolate the cuDNN benchmark and deterministic switches in four fresh process pairs; benchmark off alone controls all observed fingerprints.
- [x] Add an opt-in default-preserving CLI flag with a red/green unit test; run five-second two-process integration at two resolutions, document timing and update interview evidence. Sync only the personal fork.
