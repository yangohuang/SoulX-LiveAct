# Request-end divergence audit

The exact-audio 90-second rollout and an independent 30-second fresh-history control share the same source image, seed, canonical 720-frame BF16 audio features and deterministic cuDNN policy. Their first 662 decoded frames are pixel-identical; the final 60 frames of the fresh clip differ. The earlier exact-audio study's negative late-trend screen is unaffected, but feature equality does not explain this endpoint difference.

## Bounded attribution design

First, use the existing encoded MP4s to locate the first unequal frame and compare two H.264 encodes of **the same already-decoded 90-second source** at 722-frame versus full length. A codec-only prefix effect is evidence that output length can alter the tail of decoded video despite identical encoder input, but it cannot quantify the model's share in the original 30/90 difference.

Add a default-off `--frame_audit` diagnostic to the generator. For each block save SHA-256 of the model's actual audio-window tensor, initial seeded noise, final latent and decoded block; immediately before `export_to_video`, save SHA-256 of each RGB uint8 frame after the exact float-to-uint8 conversion that the exporter performs. Store hashes and shapes only, no feature or biometric tensors. This must leave the no-flag path unchanged.

Use the archived 5-second source WAV and image 2, seed 43, 416×720, 24 FPS, three steps, FP8 GEMM/KV and CPU offload, no anchor/pin/compile, `resident_kv_steps=1` and `--disable_cudnn_benchmark`. Make a 10-second request WAV by repeating the exact 5-second PCM samples twice. Generate a 5-second and a 10-second clip with the same source-feature tiling flag and frame audit. Verify canonical feature equality. Compare prefix block hashes, pre-encoder frame hashes and decoded MP4 pixels.

If that pair shows identical model/decoder prefix but different encoded MP4 pixels, add one **predeclared follow-up** that exercises the missing audio-end mechanism: use the exact 10-second WAV as a new canonical source, repeat it into a 20-second request, and generate 10/20-second clips with the same model settings and audit. The 10-second generator's last block has a `get_audio_emb` window beyond its 240-frame feature tensor, whereas the 20-second request has real next-period features. Predict the first mismatched block's **audio** hash is block 7; if its seeded noise still matches, a changed final latent and pre-encoder pixels there would localize the model-side endpoint effect. This short follow-up is preferable to re-running a 90-second model merely to confirm the same index behavior.

## Interpretation and stopping rule

- If pre-encoder first-five-second hashes match while MP4 tails differ, the short-run difference is encoder-only; note that the 30/90 pair still lacks archived pre-encoder hashes.
- If audio/noise/latent hashes first differ at a block, localize the earliest upstream mismatch before assigning cause. An audio-window mismatch at the short endpoint supports request-length conditioning; matched audio/noise but different latent is unresolved model/cache nondeterminism unless replicated.
- If no short-run difference appears, do not force a causal conclusion for the old 30/90 pair. Archive the negative replication and the codec-only control.

Run tests and preserve both videos, sidecars, block/frame fingerprints, decoded comparisons, resource logs and a concise report. Only sync the personal fork; no SoulX official PR.
