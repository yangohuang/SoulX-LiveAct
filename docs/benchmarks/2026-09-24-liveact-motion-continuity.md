# RTX 4090 chunk-motion continuity experiment

The 30-second, three-step RTX 4090 video visibly changes head and hand pose
near some generation boundaries even though isolated frames look good. This
report measures that symptom and keeps the proposed mitigation opt-in. The
fixture is `examples/image/1.png`, prompt `一个人在说话`, the first 5 or 30 seconds
of `examples/audio/1.wav`, 416×720 at 24 fps, seed 42, FP8 GEMM, FP8 KV,
block offload, CPU T5, and one FP8 KV step resident on the GPU.

The first generated chunk decodes to 21 frames. Each later chunk adds 32, so
adjacent-frame transitions at indices 20, 52, 84, … cross chunk boundaries.
The reported motion proxy is mean absolute grayscale pixel difference after
resizing each decoded frame to 180×104. “First 8” averages the eight
transitions starting at each boundary. It detects abrupt appearance changes;
it is not a perceptual motion or lip-sync score. Video compression and natural
gestures can also affect it.

## Diagnosis

| Controlled 5 s test | First 8-frame proxy | Observation |
| --- | ---: | --- |
| Original 3-latent VAE context | 3.329 | Same DiT latents as the two rows below |
| 4-latent VAE context | 3.222 | Small reduction |
| 5-latent VAE context | 3.189 | 4.2% below original; largest spikes remain |
| BF16 CPU KV, original VAE context | 3.41 | No improvement over FP8 KV; later blocks took about 48 s instead of 19 s |

The VAE-only variants decode the exact same generated latents. More VAE
context lowers the immediate seam difference but barely changes the first
eight frames. The BF16 KV experiment rules out FP8 history quantization as
the main cause on this fixture. These results point to a lack of explicit
motion continuity when the next chunk is sampled from fresh noise: cached
attention gives historical context, but it does not force the first new
latent to agree with the previous final latent.

## Opt-in latent anchor

`--motion_anchor_strength` blends the predicted clean latent at the
start of every new chunk toward the prior chunk's final latent at each
denoising step. The first new latent uses the requested strength; the second
uses one third of it; the other six are unchanged. The default is `0`, preserving the original
generation path. This is an inference-time experiment, not a trained model
change or a guarantee of physically correct motion.

A second 5-second test used strength 0.25, giving the second latent a weight
of about 0.083. The first-eight-frame proxy was 3.329 for the original,
2.643 at 0.25, and 2.321 at 0.45. On these short files, SyncNet's best
offset stayed -2 converted frames for all three; minimum distance was
7.695/7.576/7.891, and confidence was 6.037/6.166/5.907, respectively.
Five seconds cannot establish sustained lip sync; the 30-second comparison
below provides the longer check.

For a second 5-second check, the same settings used `examples/image/2.png`
and the first five seconds of `examples/audio/2.wav`. The original/0.25
first-eight-frame proxy was 3.808/3.103 (18.5% lower); the later-frame proxy
was 3.879/3.867. All three measured seam transitions fell. SyncNet gave
the same -2 converted-frame best offset, minimum distances 7.969/7.977,
and confidences 6.606/6.485. This is a small unfavorable score change on a
short clip; the sample is too short to establish lip-sync equivalence. The
second side-by-side artifact is
`/home/yg/yg/code/docs/liveact-4090-local/liveact-motion-second-baseline-left-anchor025-right-5s.mp4`.

An additional 5-second probe applied strength 0.45 **only at the final
denoising step** instead of at all three steps. Its first-eight-frame proxy
was 2.550, slightly below the all-step 0.25 result of 2.643, but SyncNet
minimum distance/confidence were 7.854/5.793 versus 7.576/6.166 for
all-step 0.25 and 7.695/6.037 for the original. It did not improve the
motion-versus-mouth tradeoff, so the final-step-only change was discarded
and is not part of the implementation.

A second throwaway probe used strength 0.45 but reduced its spatial weight
with a Gaussian mask around a **manually chosen mouth center**. For the first
and second 5-second fixtures, first-eight-frame proxies were 2.451 and
2.792, below the all-step 0.25 results of 2.643 and 3.103. SyncNet
minimum distance/confidence were 7.553/6.229 on the first fixture and
7.913/6.468 on the second. The second fixture's fixed mouth crop changed
*less* than with the 0.25 anchor despite the protective mask, and its
SyncNet distance/confidence moved in opposite directions relative to 0.25.
The hand-picked mask requires per-subject tuning and did not establish
reusable lip preservation. It remains outside the implementation.

| Video | First 8-frame mean | Later 24-frame mean | Seam mean | Worst seam |
| --- | ---: | ---: | ---: | ---: |
| 30 s original | 2.584 | 1.952 | 3.039 | 7.431 |
| 30 s anchor 0.25 | 2.123 | 1.949 | 2.244 | 5.097 |
| 30 s anchor 0.45 | 1.860 | 1.995 | 2.027 | 3.954 |

The three 30-second outputs all contain 722 decoded video frames at 24 fps
and 30-second audio, and FFmpeg decoded both anchored files without error.
At 0.25, 21 of 22 chunk-start windows improved and their mean fell 17.8%;
the paired-block bootstrap 95% interval for absolute improvement was
0.279–0.642 pixel values. The index-373 transition fell from 16.07 to
10.68. At 0.45, all 22
chunk-start eight-frame windows had a lower proxy and their mean fell 28.0%;
a paired block bootstrap (10,000 resamples, seed 42) gave a 95% interval of
0.565–0.903 pixel values for the absolute improvement. The sharp transition
at index 373 fell from 16.07 to 6.29, and the boundary at 244 fell from
7.43 to 3.95. The 0.45 video's highest transition is still 6.32 at
index 649, during a fast hand gesture. This improves the measured boundary
pattern but does not remove every visible discontinuity.

The subsequent-block medians were 18.954 s for the original 30-second run,
19.029 s at 0.25, and 18.964 s at 0.45. These are separate runs with the
same seed and settings, not a controlled kernel microbenchmark. The model remains far
below real-time 24 fps on this RTX 4090.

A fixed mouth-area crop had lower first-eight-frame appearance change,
5.31/4.28/3.75 for baseline/0.25/0.45, while its later-frame values were
4.96/4.94/4.93. This crop also
captures head movement, so it cannot establish whether lip motion or audio
alignment improved or worsened. A separate local SyncNet comparison used the
same 25 fps conversion and face-crop pipeline for all three 30-second files.
Each had one detected face track over 754 converted frames and the same best
offset of -2 converted frames. The baseline/0.25/0.45 minimum embedding
distances were 7.672/7.657/7.989 (lower is better), and relative confidences
were 7.079/7.012/6.599 (higher is better). At that shared offset, mean
first-eight-frame distances were 7.751/7.720/8.421. Thus 0.25 had a small,
mixed score change, while 0.45 had a clear unfavorable change at boundaries.
The latter supports a possible lip-sync tradeoff; this evaluator does
not establish an absolute delay because the evaluator changes frame rate and
has no ground-truth mouth motion. Both settings remain experimental and off by
default pending direct audiovisual review and more identities/audio.

The local review videos are
`/home/yg/yg/code/docs/liveact-4090-local/liveact-motion-baseline-left-anchor-right-5s.mp4`
and `/home/yg/yg/code/docs/liveact-4090-local/liveact-motion-baseline-left-anchor025-right-30s.mp4`.
The latter has the original on the left and 0.25 on the right; the
`liveact-motion-baseline-left-anchor-right-30s.mp4` file compares 0.45.
The shorter `liveact-motion-worst-seam-15s.gif` in the same local directory
shows the remaining abrupt movement near frame 373.
