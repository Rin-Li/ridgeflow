"""Success of the ridge walker and DWA on random maps."""
from __future__ import annotations

import argparse

import numpy as np

from ridgeflow import DWA, GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ridgeflow_rrt64.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid, device=args.device)
    walker, dwa = RidgeWalker(grid), DWA(grid)
    worlds = [sample_rect_world(np.random.default_rng(seed), grid) for seed in range(args.queries)]

    solved = {"ridge walker": [], "DWA": []}
    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occupancy, grids, starts, goals = stack(chunk, model.device)
        heatmaps = model.sample(occupancy, starts, goals, seed=begin)
        results = {
            "ridge walker": walker.walk(heatmaps, grids, starts, goals),
            "DWA": dwa.run(grids, starts, goals),
        }
        for name, result in results.items():
            solved[name] += [
                bool(ok) and world.path_free(path)
                for world, path, ok in zip(chunk, result.paths, result.reached, strict=True)
            ]

    print(f"{args.queries} random maps")
    for name, flags in solved.items():
        print(f"{name:<14}{100 * np.mean(flags):6.1f}%")


if __name__ == "__main__":
    main()
