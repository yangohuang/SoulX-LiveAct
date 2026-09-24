"""Print latent-change diagnostics from a captured LiveAct block."""

import argparse
from pathlib import Path

import torch

from boundary_probe import latent_transition_metrics


def report(probe_dir: Path, block_index: int) -> None:
    print("stage  timestep  prev-within  cross-boundary  new-within  cross/neighbor")
    for path in sorted(probe_dir.glob(f"block-{block_index}-step-*.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        metrics = latent_transition_metrics(snapshot["previous_last_two"],
                                            snapshot["current_first_two"])
        reference = (metrics["previous_within_rmse"] + metrics["current_within_rmse"]) / 2
        print(f"{snapshot['step_index']:>5}  {snapshot['timestep']:>8.1f}  "
              f"{metrics['previous_within_rmse']:>11.4f}  "
              f"{metrics['boundary_rmse']:>14.4f}  "
              f"{metrics['current_within_rmse']:>10.4f}  "
              f"{metrics['boundary_rmse']/reference:>14.3f}")

    final = torch.load(probe_dir / f"block-{block_index}-final.pt",
                       map_location="cpu", weights_only=True)
    metrics = latent_transition_metrics(final["previous_last_five"], final["current_all"])
    reference = (metrics["previous_within_rmse"] + metrics["current_within_rmse"]) / 2
    print(f"final  {'0.0':>8}  {metrics['previous_within_rmse']:>11.4f}  "
          f"{metrics['boundary_rmse']:>14.4f}  {metrics['current_within_rmse']:>10.4f}  "
          f"{metrics['boundary_rmse']/reference:>14.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("block_index", type=int)
    args = parser.parse_args()
    report(args.probe_dir, args.block_index)
