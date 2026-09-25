# Second-seed replication protocol, fixed before reviewing outputs

## Question and design

The [seed-42 pilot](2026-09-25-long-horizon-stability-study.md) found that a fixed `--motion_anchor_strength 0.45` lowers boundary-window grayscale peaks for three identities, while reducing local image change, worsening SyncNet's relative confidence, and sometimes smearing hands. This protocol tests whether those trade-offs recur at seed 43. The baseline is `--motion_anchor_strength 0`; the two arms are separate generations, not pixel-aligned counterfactuals.

For each of public example identities 1, 2 and 3, hold the reference PNG, first 30 seconds of its WAV, Chinese prompt `一个人在说话`, model checkpoint, 416×720 output, 24 FPS, three denoising steps, FP8 GEMM/KV, CPU block offload, resident KV step count 1 and seed 43 fixed between arms. The six output videos must decode to 722 frames and have identical video geometry and rate. Source PNG and WAV hashes, trimmed WAV hashes, output hashes, software commands and any failure/retry are part of the record. The 4090 is used sequentially to avoid VRAM contention.

## Endpoints and interpretation rule

The primary descriptive endpoint is the mean peak in the first eight adjacent-frame grayscale-MAE transitions at all 22 chunk boundaries (`20+32k`). Report each identity × seed × arm, with the within-pair relative change. Secondary motion endpoints are maximum peak, eight-transition change sum, delayed-peak count and early/middle/late thirds. A lower grayscale peak alone does not mean better action quality; review the three largest baseline-peak windows and three largest anchor-peak windows for each identity, including hand clarity and motion timing.

At one frame per second, the local CPU InsightFace evaluator records single-face coverage and reference/first-face cosine curves. Its 90% overall/early/late coverage rule must pass before interpreting similarity. It is a pose-sensitive proxy, not a validated identity-drift metric. The local SyncNet S3FD pipeline uses the same model and 25-FPS conversion for both arms. Check converted face-track length, audio samples, best offset, minimum distance and confidence; if mux metadata causes different track lengths, compare matched endpoints and disclose the correction. Visual lip and action review remains necessary.

The fixed anchor is **not** a quality success unless its peak reduction holds across seeds *without* a repeated motion-suppression/ghosting signal or a repeated unfavorable lip-sync signal. The seed-42 result already fails that provisional gate; seed 43 tests robustness of the diagnosis rather than attempting to rescue the claim. If seed 43 confirms the trade-off, reject fixed anchoring as the algorithmic candidate and prioritize a measured, selective mechanism only after defining a new held-out test. If results conflict, report heterogeneity and do not tune thresholds on these six cases.

## Runtime and decision boundary

Existing 4090 measurements put later chunks around 19 seconds per 32 output frames (about 1.7 output FPS), so this is a quality study rather than a real-time demonstration. A future selective inference candidate should measure additional DiT passes, GPU memory and seconds per chunk alongside video quality; a reduced-resolution/two-step acceleration choice needs matched quality evidence. No candidate is submitted upstream based on grayscale MAE alone, and no 30-second result is called infinite-video stability.
