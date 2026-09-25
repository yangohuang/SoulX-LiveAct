# Long-horizon stability study design

## Purpose

Treat SoulX-LiveAct as a reproducible streaming-video testbed, rather than making repair of one frame-373 pose jump the research objective. The first study asks whether an inference intervention actually reduces abrupt motion over a 30-second rollout, or merely delays or suppresses motion. Its result must be useful even if no intervention passes a quality gate.

This maps to the Vivix JD's long-form generation, error accumulation, temporal consistency, inference cost, and independent experiment design. The [Vivix-W1 report](https://vivix.ai/tech-report-vivix-w1) describes these as system goals; this study does not claim to implement Vivix-W1 or its training method.

## Fixed cohort and paired arms

Begin with three existing 416×720, 24 FPS, 722-frame videos from one image, audio, prompt and seed: no anchor, anchor strength 0.25, anchor strength 0.45. The initial model, denoising steps and memory strategy are the same. These are three separately generated runs and are not pixel-identical; comparisons are descriptive rather than paired-frame counterfactuals. A later replication gate requires at least three identities and two seeds before claiming generality.

The first video has 21 decoded frames; each later chunk adds 32. Therefore the cross-chunk transition indices are 20, 52, 84, and so on. Decode each video using FFmpeg with `-vsync 0`, scale to 180×104, and compute grayscale mean absolute change for each adjacent pair. Refuse mismatched frame counts, dimensions or frame rate.

## Outcomes and safeguards

For every boundary, record (a) immediate transition, (b) maximum change among the first eight transitions and its offset, and (c) total change over those eight transitions. Summarize the 22 boundary windows overall and by early/middle/late thirds. If the immediate transition falls but the maximum remains or moves later, label the pattern **displacement**, not a continuity improvement. If total change falls without a reference motion trajectory, label it **possible motion suppression**, not improvement. The within-chunk transition distribution is context, not a ground-truth target.

Keep identity and audio alignment separate: the first CPU frame-change metric does not measure either. Reuse the existing same-pipeline 30-second SyncNet comparison as an independent lip-sync guardrail. A separate local InsightFace face-reference curve may be recorded with detector coverage, but remains a proxy until multiple identities and visual checks are available. Include frame samples around the largest event for human review. Report wall time/FPS from existing logs as inference cost, without implying real-time performance.

## Deliverables and decision

Deliver a reproducible CPU evaluator, focused tests, JSON results, a short research report and a Vivix interview evidence paragraph. The evaluator must not alter the model or generation path. A genuine method claim later requires replicated video-level benefits on multiple identities/seeds without degrading visible motion, identity or lip sync. The current study can conclude that metrics are insufficient or that a candidate fails; those are valid research findings. Keep it on a personal research branch, separate from the upstream inference PR candidate.
