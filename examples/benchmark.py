"""Success of the ridge walker and DWA on random maps.

ridge walker      the backtracking walker on the sampled heatmap
+ projection      the heatmap first projected so a free route has H >= 0.5 all along it
                  (``routes.project``); the walker then reads only ridge near that route
DWA               the classic dynamic window approach, best of 60 weight settings
A path counts when it reaches the goal and clears every obstacle rectangle exactly.
"""

from __future__ import annotations

import argparse

import numpy as np

from ridgeflow import DWA, GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack
from ridgeflow.routes import project


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

    solved = {"ridge walker": [], "+ projection": [], "DWA": []}
    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occupancy, grids, starts, goals = stack(chunk, model.device)
        heatmaps = model.sample(occupancy, starts, goals, seed=begin)
        projected, route = project(heatmaps, ~grids, grid.cells(starts), grid.cells(goals))
        results = {
            "ridge walker": walker.walk(heatmaps, grids, starts, goals),
            "+ projection": walker.walk(projected, grids, starts, goals, route=route),
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
