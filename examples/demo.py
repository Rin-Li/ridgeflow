"""Render the figures in the README."""
from __future__ import annotations

import argparse

import numpy as np

from ridgeflow import DWA, SCENES, GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack
from ridgeflow.render import Panel, figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ridgeflow_rrt64.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--outdir", default="assets")
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid, device=args.device)
    walker, dwa = RidgeWalker(grid), DWA(grid)

    def solve(worlds, seed=0):
        occupancy, grids, starts, goals = stack(worlds, model.device)
        heatmaps = model.sample(occupancy, starts, goals, seed=seed)
        ours = walker.walk(heatmaps, grids, starts, goals)
        theirs = dwa.run(grids, starts, goals)
        return heatmaps.cpu().numpy(), ours, theirs

    world = sample_rect_world(np.random.default_rng(3), grid)
    heatmaps, ours, _ = solve([world])
    path, ok, residual = ours.paths[0], bool(ours.reached[0]), ours.residual[0].cpu().numpy()
    figure(
        [
            Panel("sampled ridge H", world, heatmaps[0]),
            Panel("walk", world, heatmaps[0], walker=(path, ok)),
            Panel("residual R after the walk", world, residual, walker=(path, ok)),
        ],
        f"{args.outdir}/method.png",
    )

    panels = []
    for name, make in SCENES.items():
        world = make(grid)
        heatmaps, ours, theirs = solve([world] * args.samples)
        solved = ours.reached.nonzero()[0]
        shown = int(solved[0]) if len(solved) else 0
        panels.append(
            Panel(
                name,
                world,
                heatmaps[shown],
                walker=(ours.paths[shown], bool(ours.reached[shown])),
                dwa=(theirs.paths[0], bool(theirs.reached[0])),
            )
        )
    figure(panels, f"{args.outdir}/traps.png")

    worlds = [sample_rect_world(np.random.default_rng(seed), grid) for seed in range(6)]
    heatmaps, ours, theirs = solve(worlds)
    figure(
        [
            Panel(
                f"random map {i}",
                w,
                heatmaps[i],
                walker=(ours.paths[i], bool(ours.reached[i])),
                dwa=(theirs.paths[i], bool(theirs.reached[i])),
            )
            for i, w in enumerate(worlds)
        ],
        f"{args.outdir}/random.png",
        columns=3,
    )


if __name__ == "__main__":
    main()
