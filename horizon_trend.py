"""Summarize fixed-duration portions of a causal rollout's boundary windows."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


def analyze_horizon(windows: list[dict], fps: int, segment_seconds: int = 30,
                    segment_count: int = 3) -> dict:
    if fps < 1 or segment_seconds < 1 or segment_count < 2 or not windows:
        raise ValueError("need windows and a positive multi-segment time layout")
    span = fps * segment_seconds
    groups = [[] for _ in range(segment_count)]
    for window in windows:
        index = int(window["index"])
        if index < 0:
            raise ValueError("negative boundary index")
        slot = index // span
        if slot < segment_count:
            groups[slot].append(window)
    if any(not group for group in groups):
        raise ValueError("every requested time segment needs a complete boundary window")
    early_p95 = float(np.percentile([w["peak"] for w in groups[0]], 95))
    segments = []
    for slot, group in enumerate(groups):
        segments.append({
            "start_second": slot * segment_seconds,
            "end_second": (slot + 1) * segment_seconds,
            "count": len(group),
            "median_peak": float(median(w["peak"] for w in group)),
            "mean_peak": float(mean(w["peak"] for w in group)),
            "mean_immediate": float(mean(w["immediate"] for w in group)),
            "mean_motion_sum": float(mean(w["motion_sum"] for w in group)),
            "severe_count": sum(w["peak"] > early_p95 for w in group),
        })
    first, last = segments[0], segments[-1]
    if first["mean_motion_sum"] <= 0 or first["median_peak"] <= 0:
        raise ValueError("early segment has no measurable movement")
    motion_ratio = last["mean_motion_sum"] / first["mean_motion_sum"]
    peak_ratio = last["median_peak"] / first["median_peak"]
    motion_retained = motion_ratio >= 0.9
    worsening = peak_ratio >= 1.25
    return {"fps": fps, "segment_seconds": segment_seconds,
            "severe_threshold_early_p95": early_p95,
            "segments": segments, "late_to_early_median_peak_ratio": peak_ratio,
            "late_to_early_motion_ratio": motion_ratio,
            "motion_retained": motion_retained,
            "deterioration_signal": bool(motion_retained and worsening)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--segment-count", type=int, choices=(2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw_json.read_text())
    video = raw["videos"][args.label]
    fps_num, fps_den = map(int, video["metadata"]["fps"].split("/"))
    if fps_den != 1:
        raise ValueError("expected integer FPS")
    result = analyze_horizon(video["boundaries"], fps=fps_num,
                             segment_count=args.segment_count)
    result.update({"source": video["source"], "source_sha256": video["sha256"],
                   "raw_json": str(args.raw_json), "label": args.label})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
