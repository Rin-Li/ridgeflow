"""Train the ridge model with rectified flow matching."""
from __future__ import annotations

import argparse
from pathlib import Path

from ridgeflow import GridSpec
from ridgeflow.training import RidgeDataset, TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="pickled dict of map/start/goal/paths")
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--output-dir", default="checkpoints/run")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--extent", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    grid = GridSpec(size=args.grid_size, extent=args.extent)
    dataset = RidgeDataset(args.dataset, grid)
    val_dataset = RidgeDataset(args.val_dataset, grid) if args.val_dataset else None

    config = TrainConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
    )
    config.model["base_channels"] = args.base_channels

    print(f"{len(dataset)} training queries on a {grid.size}x{grid.size} grid")
    best = train(dataset, config, val_dataset, grid, device=args.device)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
