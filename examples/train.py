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
    parser.add_argument(
        "--gap-weight", type=float, default=0.0, help="gap loss weight (100 in the paper; 0: off)"
    )
    parser.add_argument(
        "--geodesic-target",
        action="store_true",
        help="ridge distance measured in free space (no leaking through walls)",
    )
    parser.add_argument(
        "--augment", action="store_true", help="random symmetry of the square per query"
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--ridge-sigma", type=float, default=1.5, help="0: binary raster target")
    parser.add_argument("--seed", type=int, default=0, help="init, noise, shuffling, augmentation")
    args = parser.parse_args()

    grid = GridSpec(size=args.grid_size, extent=args.extent)
    dataset = RidgeDataset(
        args.dataset, grid, args.ridge_sigma, args.geodesic_target, augment=args.augment
    )
    val_dataset = (
        RidgeDataset(args.val_dataset, grid, args.ridge_sigma, args.geodesic_target)
        if args.val_dataset
        else None
    )

    config = TrainConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        gap_weight=args.gap_weight,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    config.model["base_channels"] = args.base_channels

    print(f"{len(dataset)} training queries on a {grid.size}x{grid.size} grid")
    best = train(dataset, config, val_dataset, grid, device=args.device)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
