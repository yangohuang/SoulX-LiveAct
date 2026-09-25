# Repeated-speech history test plan

- [x] Construct and verify sample-identical 30+30+30-second speech and fresh 30-second requests; freeze hashes and generation settings.
- [x] Generate the 90-second long-history video on the single RTX 4090 and reuse the verified archived independent 30-second same-audio control, checking full decode and resource use.
- [x] Audit post-SoX Wav2Vec conditioning equality, then evaluate three repeated periods with the same boundary, face and SyncNet pipeline; compare fresh control, matched visual lip moments and preregistered gate. Conditioning and lip-score gates failed, so seed 44 was not run.
- [x] Save runnable code/JSON/video/figures and clear inference limits; update roadmap, interview evidence and memory; run tests, push only personal fork and verify remote SHA.
