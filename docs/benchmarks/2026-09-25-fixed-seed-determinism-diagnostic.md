# Fixed seed was not enough for cross-process LiveAct comparisons

**Finding:** On this RTX 4090 setup, independently launched three-step LiveAct runs with identical image, audio, seed and model caches diverged **before DiT**. Disabling cuDNN benchmarking made the observed 224×384 and 416×720 repeated runs byte-identical, with no material 416×720 steady-block slowdown in this five-second check. The [diagnostic specification](../superpowers/specs/2026-09-25-fixed-seed-determinism-diagnostic.md), [fingerprint wrapper](../../diagnose_fixed_seed.py), [summary JSON](artifacts/2026-09-25-fixed-seed-diagnostic/summary.json), eight two-block fingerprint records and five 5-second videos preserve the evidence. This is an **opt-in reproducibility control**, not a claim of universal determinism or better generation quality.

## First divergence, not just an MP4 mismatch

The input was image `examples/image/1.png`, 1.5-second PCM WAV, prompt `一个人在说话`, seed 42, 224×384/24 FPS, three steps, cached FP8 DiT/T5, FP8 CPU KV with one resident step, block offload and no compile or pinning. Two independent default processes (`a`, `b`) each generated two blocks. The wrapper fingerprints tensors at preprocessing and generation boundaries without changing the tensors. Source hashes and exact commands are in the [summary](artifacts/2026-09-25-fixed-seed-diagnostic/summary.json); the linked [run-a fingerprints](artifacts/2026-09-25-fixed-seed-diagnostic/liveact-determinism-a-fingerprints.json) are one of eight archived records. Both default videos decode fully, but their MP4 SHA-256 values differ.

| Boundary in causal order | Default cross-process result | Meaning |
|---|---|---|
| Transformed reference image, CLIP context, resampled audio | Exact tensor hashes equal | Source media and these preprocessing outputs are matched |
| Wav2Vec embedding | **First differing fingerprint**; MAE 0.001900, relative RMSE 1.26% | Audio model condition differs before the generator |
| Reference-image VAE latent | Also differs; MAE 0.0000113, relative RMSE 0.025% | Independent image-condition path differs before the generator |
| First generated noise, CLIP/context tensors | Exact hashes equal | CLI seed controls the sampled noise; the seed itself is not lost |
| First DiT output | Differs; MAE 0.03049, relative RMSE 3.70% | Early condition differences propagate and amplify; DiT kernel variation is not separately excluded |
| First VAE decode output | Differs; MAE 0.01667, relative RMSE 9.71% | Visible output already differs in the first block |

The active LightVAE `encode` returns the mean latent, so this is **not** explained by sampling a VAE posterior before `torch.manual_seed`. `generate.py` set `torch.backends.cudnn.benchmark=True` by default. It is an inference from the factorial experiment below that cuDNN's benchmark-based algorithm choice is the controlling factor here; we did not log internal cuDNN algorithm IDs or prove that every kernel is deterministic.

## Factorial switch test

Each policy was tested in two fresh independent two-block processes. “Same” means **all 35 tensor boundary hashes** and the final MP4 SHA-256 match within that pair.

| cuDNN benchmark | cuDNN deterministic | Pair | First differing boundary | MP4 bytes |
|---|---|---|---|---|
| On (current default) | Off | `a` / `b` | Wav2Vec embedding | Different |
| Off | On | `c` / `d` | None | **Same** |
| Off | Off | `e` / `f` | None | **Same** |
| On | On | `g` / `h` | Wav2Vec embedding | Different |

Thus disabling **benchmark alone** was sufficient in these runs; requiring `cudnn.deterministic=True` while leaving benchmark on was not. The implementation adds an opt-in `--disable_cudnn_benchmark` flag, applied before loading/encoding models. Default behavior remains unchanged. The [unit test](../../tests/test_cudnn_reproducibility.py) checks that the flag disables benchmark without silently changing the deterministic setting, while the 4090 integration videos check the effect end to end.

## End-to-end reproduction and cost

Two independent five-second 224×384 runs with the real CLI flag produced the same MP4 SHA-256 (`54375fcfeee6cf741f6226999af4e7f848388d600eb8f1aeaea34089be31c5c6`). The two 416×720 CLI runs also matched byte-for-byte (`013f26f14b07b0cb1fbfb0408541b681b0a7b173cd5aab3338ca8d90217577bc`). Each decoded to 117 video frames and 4.864 seconds of encoded audio. The [default 416×720 video](artifacts/2026-09-25-fixed-seed-diagnostic/liveact-cudnn-default-416.mp4), [flagged 416×720 video](artifacts/2026-09-25-fixed-seed-diagnostic/liveact-cudnn-off-416-a.mp4), and [six-timepoint sheet](artifacts/2026-09-25-fixed-seed-diagnostic/liveact-cudnn-416-contact.png) show similar visible content, but the two policies intentionally need not sample the exact same pixels. Visual similarity does not prove lip quality.

| 5-s run, three steps | Later block costs, s | Median, s | Difference from fresh default |
|---|---|---:|---:|
| 224×384 default, previous two same-arm runs | 7.451/7.267/7.188 and 7.297/7.531/7.101 | 7.267 / 7.297 | Reference range; different launch period |
| 224×384 benchmark off, run A | 7.766/7.203/7.455 | 7.455 | About 2.6% above earlier 7.267 reference |
| 224×384 benchmark off, run B | 7.729/7.171/7.414 | 7.414 | About 2.0% above earlier 7.267 reference |
| 416×720 fresh default | 19.059/18.977/18.955 | **18.977** | — |
| 416×720 benchmark off, run A | 18.816/19.061/19.025 | **19.025** | **+0.25%** |
| 416×720 benchmark off, run B | 18.832/19.086/19.059 | **19.059** | **+0.43%** |

The 416×720 medians imply about 1.69 generated FPS for either policy, nowhere near 24 FPS. These are only three steady blocks per run; the small timing differences are not statistical proof of zero cost. The measured benefit is experimental reproducibility. A future official-quality or speed PR would need longer, held-out audiovisual checks; this change remains on the personal fork pending user review.

**Implication for prior A/Bs:** A shared seed did not guarantee matched Wav2Vec and VAE conditioning across default-process launches. Earlier two-step/three-step and anchor quality results still describe their observed videos and scores, but the single-pair estimates should not be treated as numerically controlled counterfactuals. The 0.612 SyncNet two-step loss remains an observed gate failure; a deterministic repeat is required before asserting that its full size is caused by the removed denoising step. Use `--disable_cudnn_benchmark` consistently in future paired experiments.
