"""Join opt-in pre-output latent signals to decoded LiveAct boundary motion."""

import argparse
import json
import math
from pathlib import Path

from scipy.stats import spearmanr

from long_horizon_stability import study_video


def correlate_signal_with_motion(signal: dict, motion: dict, top_k: int = 5) -> dict:
    """Validate block/frame alignment and compare final-step signals to motion peaks."""
    blocks = signal["blocks"]
    windows = motion["boundaries"]
    if len(blocks) != len(windows) or not blocks:
        raise ValueError("signal and motion boundary counts differ")
    for block, window in zip(blocks, windows):
        index = window["index"]
        if block["boundary_transition"] != [index, index + 1]:
            raise ValueError("signal/video boundary alignment differs")
    rows = []
    for block, window in zip(blocks, windows):
        first = block["steps"][0]["metrics"]
        final = block["steps"][-1]["metrics"]
        rows.append({"block_index": block["block_index"],
                     "transition": window["index"],
                     "peak": window["peak"],
                     "immediate": window.get("immediate"),
                     "peak_offset": window.get("peak_offset"),
                     "first_step": first, "final_step": final})
    peak_order = sorted(rows, key=lambda row: (-row["peak"], row["block_index"]))
    result = {"count": len(rows), "rows": rows,
              "top_peak_blocks": [row["block_index"] for row in peak_order[:top_k]]}
    metric_names = sorted(set.intersection(*(set(row["final_step"]) for row in rows)))
    for name in metric_names:
        order = sorted(rows, key=lambda row: (-row["final_step"][name], row["block_index"]))
        top_blocks = [row["block_index"] for row in order[:top_k]]
        association = float(spearmanr([row["final_step"][name] for row in rows],
                                      [row["peak"] for row in rows]).statistic)
        immediate = None
        if all(row["immediate"] is not None for row in rows):
            immediate_value = float(spearmanr([row["final_step"][name] for row in rows],
                                              [row["immediate"] for row in rows]).statistic)
            immediate = immediate_value if math.isfinite(immediate_value) else None
        result[name] = {"spearman_peak": association if math.isfinite(association) else None,
                        "spearman_immediate": immediate,
                        "top_blocks": top_blocks,
                        "top_k_overlap": sorted(set(top_blocks) &
                                                set(result["top_peak_blocks"]))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal = json.loads(args.signal.read_text())
    motion = study_video(args.video)
    result = correlate_signal_with_motion(signal, motion)
    result["signal_path"] = str(args.signal.resolve())
    result["video"] = {key: motion[key] for key in ("source", "sha256", "metadata", "summary")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "video")}, indent=2))


if __name__ == "__main__":
    main()
