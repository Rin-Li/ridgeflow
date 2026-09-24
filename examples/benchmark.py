"""Success and path length of the ridge walker, DWA and RRT* on random maps."""
from __future__ import annotations

import argparse

import numpy as np

from ridgeflow import DWA, GridSpec, RidgeModel, RidgeWalker, RRTStar, sample_rect_world, stack


def length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


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

    paths = {"ridge walker": [], "DWA": [], "RRT*": []}
    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occupancy, grids, starts, goals = stack(chunk, model.device)
        heatmaps = model.sample(occupancy, starts, goals, seed=begin)
        ours = walker.walk(heatmaps, grids, starts, goals)
        theirs = dwa.run(grids, starts, goals)
        for world, path, ok in zip(chunk, ours.paths, ours.reached, strict=True):
            paths["ridge walker"].append(path if ok and world.path_free(path) else None)
        for world, path, ok in zip(chunk, theirs.paths, theirs.reached, strict=True):
            paths["DWA"].append(path if ok and world.path_free(path) else None)
    paths["RRT*"] = [RRTStar(w, rng=seed).plan() for seed, w in enumerate(worlds)]

    reference = paths["RRT*"]
    print(f"{args.queries} random maps\n")
    print(f"{'method':<14}{'success':>9}{'length / RRT*':>16}")
    for name, found in paths.items():
        success = np.mean([p is not None for p in found])
        ratios = [
            length(p) / length(r) for p, r in zip(found, reference, strict=True)
            if p is not None and r is not None
        ]
        print(f"{name:<14}{100 * success:>8.1f}%{np.mean(ratios):>16.3f}")


if __name__ == "__main__":
    main()
