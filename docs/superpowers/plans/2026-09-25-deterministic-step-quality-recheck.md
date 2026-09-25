# Deterministic step-quality recheck plan

- [x] Freeze exact audio, image, request files and three-step/two-step commands with the cuDNN reproducibility flag in both arms.
- [x] Generate fresh seed43 30-second videos on the 4090; verify complete decode, block timings and hardware samples.
- [x] Recompute boundary motion, matched SyncNet, face proxy and visual worst windows; apply the original quality gate.
- [x] Seed43 lost 0.667 SyncNet confidence, so run paired seed44 and all same quality checks; seed44 lost 0.430.
- [x] Archive media, JSON and report; update roadmap/interview evidence, run tests and sync only personal fork.
