# Where did the seed-44 boundary-motion deficit come from?

The [first-audio two-step experiment](2026-09-25-two-step-audio-conditioning.md) passed the lip-sync proxy on two deterministic seeds, but seed 44 retained only **78.2%** of the three-step control's full-frame grayscale change in the 22 preregistered eight-transition boundary windows. That original multi-objective gate **still fails**. This is a post hoc, CPU-only source audit of the same six archived videos, asking whether the loss is specifically hand/body motion, a broader image change, or a change in when motion occurs. It adds diagnostics, not a replacement acceptance threshold.

## Fixed regions and checks

[The spatial audit script](../../spatial_motion_audit.py) decodes all six 722-frame videos at 24 FPS to 104×180 grayscale. It measures adjacent-frame mean absolute difference (MAE) and Farnebäck flow magnitude in a whole-frame mask, mouth, upper head, lower-body/hand band and two outer-background corners. All rectangles were frozen in the [design](../superpowers/specs/2026-09-25-motion-source-audit.md) before comparing outcomes; the [region overlay](artifacts/2026-09-25-spatial-motion-source-audit/region-overlay.png) shows where they land on frame 480. Mouth and head overlap, so their values cannot be summed. The [compressed raw six-video JSON](artifacts/2026-09-25-spatial-motion-source-audit/six-video-spatial-motion.json.gz) records each of the 721 transitions, 22 complete boundary windows, source hashes and exact box coordinates. A synthetic moving-square test confirms that lower-band movement is detected without leaking into stationary upper/background masks.

The resized whole-frame boundary MAE ratio reproduces the original evaluator's direction: first-audio two-step / three-step is **1.014** at seed 43 and **0.786** at seed 44, close to the original 180×104 ratios **1.009** and **0.782**. Thus the failure is not a mistake in extracting the chunk indices or an artefact of only one resize orientation. Values below are candidate/control ratios on identical transition sets; boundary windows contain 176 transitions, all other times 545.

| Seed and region | Boundary MAE | Outside MAE | Boundary flow | Outside flow |
|---|---:|---:|---:|---:|
| 43, whole frame | 1.014 | 0.988 | 1.127 | 1.055 |
| 43, lower body/hands | 1.174 | 1.099 | 1.332 | 1.175 |
| 44, whole frame | **0.786** | 0.942 | **0.844** | 1.034 |
| 44, mouth | 0.804 | 0.883 | 0.783 | 0.857 |
| 44, upper head | 0.758 | 0.839 | 0.774 | 0.856 |
| 44, lower body/hands | **0.809** | **1.014** | **0.886** | **1.136** |
| 44, outer background | 0.708 | 0.755 | 0.659 | 0.723 |

The [region chart](artifacts/2026-09-25-spatial-motion-source-audit/spatial-boundary-outside-ratios.png) makes two points: the seed-44 deficit affects the head and even background corners, so a whole-frame pixel-change score cannot be read as a pure hand-motion score; and lower-body flow is lower **near boundaries** but higher outside them. The new seed-44 arm is below its three-step control in 13/22 lower-body MAE windows and 12/22 lower-body flow windows. Such counts and whole-video means suggest a change in motion timing/distribution rather than a uniform body-motion freeze. They do not establish natural gestures, since optical flow also responds to texture changes and inaccurate correspondences.

The stock two-step control matters: at seed 44 its lower-body boundary MAE/flow ratios are **0.931/0.920** versus three steps, while the first-audio arm is **0.809/0.886**; at seed 43 stock two-step is **1.193/1.179** and first-audio is **1.174/1.332**. The direction of the body-band signal differs by seed, so one random rollout should not determine a schedule choice.

## Pose triangulation and limits

Because the spatial bands still mix arms, torso and background, an exploratory addendum used [MediaPipe Pose](../../pose_motion_audit.py) on all six videos. [Frame 480 with landmarks](artifacts/2026-09-25-spatial-motion-source-audit/pose-frame480.png) shows a single sampled difference in arm position, not a trajectory conclusion. The [compressed raw pose JSON](artifacts/2026-09-25-spatial-motion-source-audit/six-video-pose.json.gz) stores per-frame normalized coordinates and visibility. Displacement comparisons use only adjacent transitions where **all three same-seed arms** have visibility ≥0.5 and in-frame points, so missing detections cannot silently count as zero motion.

For seed 44, common support covers all 176 boundary transitions for the left wrist and 173/176 for the right wrist. First-audio / three-step mean wrist displacement is **0.977** on the left and **1.207** on the right at boundaries; outside windows the ratios are **0.927** and **1.122**. On 695/722 frames where both wrists are jointly valid in all three arms, mean horizontal wrist separation is 0.580 for three steps and 0.605 for first-audio two steps. These measurements do **not** support a uniform loss of tracked wrist travel. They are descriptive: the generated hands can be partially cropped, Pose elbows have poor joint common support (0/176 boundary transitions for one seed-44 elbow), and image-space tracker jitter can inflate path length. A stronger body-motion quality judgment would need a validated detector or human review across the full videos.

## Research consequence

Keep the original **failed** seed-44 motion gate and the rejection of first-audio two steps as a default. Refine the causal wording: it reduces *boundary-local, whole-frame image change* on seed 44, while the exploratory lower-body flow and wrist trajectories do not prove overall gesture suppression. The improved SyncNet despite lower mouth-region change also illustrates that lip **timing** and lip **motion amplitude** are different measurements. For a Vivix-style long-form/step-acceleration project, this audit strengthens the evaluation story: it exposes where a convenient proxy conflates face, body and background, and prevents calling a smaller boundary peak an unqualified continuity win. The next algorithmic candidate should be evaluated with a preregistered body/pose metric on fresh identities, alongside SyncNet and human video review, before any upstream quality PR.
