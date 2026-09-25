# Request-end divergence implementation plan

- [x] Archive the original 30/90 first-difference record and codec-only same-source truncated/full control with hashes, framewise summary and testable interpretation.
- [x] Add a failing unit test for exact exporter-input uint8 frame hashes and first-divergence classification, then implement the pure helpers.
- [x] Add a default-off `--frame_audit` generation flag that fingerprints block audio, initial noise, final latent, decoded block and the RGB uint8 frames immediately before H.264 encoding; cover no-flag behavior and syntax.
- [x] Freeze a 5-second PCM source and exact repeated 10-second WAV; run 5/10-second 4090 pair with identical model settings and auditable feature hashes.
- [x] After checking the 5/10 prefix, run the predeclared 10/20-second audio-clamp follow-up if the 5/10 model prefix matches while encoded pixels differ.
- [x] Compare block input/output, pre-encoder and post-encoder prefix hashes; report what is identified and what the original 30/90 artifacts cannot resolve.
- [x] Archive videos, JSON/logs and hash manifest; update Vivix evidence boundaries. Run relevant tests and sync only the personal fork, verifying remote/local SHA and clean worktree.
