"""Train a small latent bridge from frozen LiveAct rollout snapshots."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rollout_bridge import RolloutBridge, make_pseudo_boundary


def load_blocks(directories: list[Path]) -> list[torch.Tensor]:
    """Load per-block generated latents without crossing actual boundaries."""
    blocks = []
    for directory in directories:
        paths = sorted(directory.glob("block-*.pt"))
        if not paths:
            raise ValueError(f"no rollout blocks in {directory}")
        for path in paths:
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            latent = snapshot["latent"]
            if latent.ndim != 4 or latent.shape[1] < 5:
                raise ValueError(f"invalid latent shape in {path}")
            blocks.append(latent.float())
    return blocks


def draw_batch(blocks: list[torch.Tensor], batch_size: int, seed: int,
               identity_only: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample deterministic synthetic boundary examples from generated blocks."""
    if not blocks or batch_size < 1:
        raise ValueError("blocks must be nonempty and batch size positive")
    rng = random.Random(seed)
    examples = []
    for _ in range(batch_size):
        block = rng.choice(blocks)
        seam = rng.randint(2, block.shape[1] - 3)
        mix = 0.0 if identity_only or rng.random() < 0.2 else rng.uniform(0.35, 1.0)
        donor_shift = rng.choice((-4, -3, 3, 4))
        examples.append(make_pseudo_boundary(block, seam, mix, donor_shift))
    return tuple(torch.stack([example[index] for example in examples]) for index in range(3))


@torch.no_grad()
def evaluate_bridge(model: RolloutBridge, blocks: list[torch.Tensor],
                    sample_count: int, seed: int) -> dict[str, float]:
    """Compare correction error with uncorrected input and clean-input drift."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    model_errors, baseline_errors, clean_drifts = [], [], []
    for index in range(sample_count):
        history, current, target = draw_batch(blocks, 1, seed + index)
        history, current, target = (x.to(device) for x in (history, current, target))
        corrected = model(history, current)[:, :, :2]
        model_errors.append(float(F.l1_loss(corrected, target)))
        baseline_errors.append(float(F.l1_loss(current[:, :, :2], target)))
        clean_history, clean_current, _ = draw_batch(blocks, 1, seed + 100000 + index,
                                                     identity_only=True)
        clean_history, clean_current = clean_history.to(device), clean_current.to(device)
        clean_corrected = model(clean_history, clean_current)
        clean_drifts.append(float(F.l1_loss(clean_corrected[:, :, :2], clean_current[:, :, :2])))
    model.train(was_training)
    return {"model_l1": sum(model_errors) / sample_count,
            "uncorrected_l1": sum(baseline_errors) / sample_count,
            "identity_drift_l1": sum(clean_drifts) / sample_count}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_blocks = load_blocks(args.train_dir)
    validation_blocks = load_blocks(args.validation_dir)
    channels = train_blocks[0].shape[0]
    if any(block.shape[0] != channels for block in train_blocks + validation_blocks):
        raise ValueError("all latent blocks must have the same channel count")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RolloutBridge(channels=channels, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    initial = evaluate_bridge(model, validation_blocks, args.validation_samples, args.seed + 50000)
    best_score = float("inf")
    best_state = None
    history = []
    start = time.monotonic()

    for step in range(args.steps):
        history_batch, current, target = draw_batch(train_blocks, args.batch_size,
                                                    args.seed + step)
        history_batch, current, target = (x.to(device) for x in (history_batch, current, target))
        optimizer.zero_grad(set_to_none=True)
        corrected = model(history_batch, current)
        loss = F.l1_loss(corrected[:, :, :2], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            score = evaluate_bridge(model, validation_blocks, args.validation_samples,
                                    args.seed + 50000)
            score.update({"step": step + 1, "train_l1": float(loss.detach())})
            history.append(score)
            print(json.dumps(score), flush=True)
            selection = score["model_l1"] + 0.2 * score["identity_drift_l1"]
            if selection < best_score:
                best_score = selection
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate_bridge(model, validation_blocks, args.validation_samples,
                            args.seed + 50000)
    result = {"train_directories": [str(path) for path in args.train_dir],
              "validation_directories": [str(path) for path in args.validation_dir],
              "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
              "hidden": args.hidden, "learning_rate": args.lr,
              "train_block_count": len(train_blocks),
              "validation_block_count": len(validation_blocks),
              "initial_validation": initial, "final_validation": final,
              "history": history, "wall_seconds": time.monotonic() - start,
              "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated()
                                            if device.type == "cuda" else 0)}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "channels": channels,
                "hidden": args.hidden, "result": result}, args.checkpoint)
    args.checkpoint.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"checkpoint": str(args.checkpoint), "initial": initial,
                      "final": final, "wall_seconds": result["wall_seconds"],
                      "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--validation_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--validation_samples", type=int, default=32)
    train(parser.parse_args())
