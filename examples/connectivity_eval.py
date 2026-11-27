"""Raw ridge connectivity, walker success and route diversity of one or more checkpoints."""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack
from ridgeflow.model.targets import path_heatmap, world_to_pixel
from ridgeflow.routes import bottleneck, hedging
from ridgeflow.training.gap import ON_RIDGE, geodesic_mask

sys.path.insert(0, str(Path(__file__).parent))




def diversity(masks: np.ndarray) -> float:
    """Mean pairwise 1 - IoU of ``(K, S, S)`` boolean ridge masks."""
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = (masks[i] | masks[j]).sum()
            vals.append(1.0 - (masks[i] & masks[j]).sum() / max(union, 1))
    return float(np.mean(vals))


def _rrt(args):
    from make_dataset import RRTStar

    torch.set_num_threads(1)
    seed, k = args
    grid = GridSpec()
    world = sample_rect_world(np.random.default_rng(seed), grid)
    out = []
    for j in range(k):
        p = RRTStar(world, rng=1000 * seed + j).plan()
        if p is not None:
            h = path_heatmap(world_to_pixel(p, grid), grid.size)[0].T.numpy()
            out.append(h >= ON_RIDGE)
    return diversity(np.stack(out)) if len(out) > 1 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rrt-maps", type=int, default=200, help="RRT* diversity reference")
    args = parser.parse_args()

    grid = GridSpec()
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]

    if args.rrt_maps:
        with Pool(20) as pool:
            ref = pool.map(_rrt, [(s, args.k) for s in range(args.rrt_maps)])
        print(
            f"RRT* demonstrations: route diversity {np.nanmean(ref):.3f} "
            f"(1-IoU, {args.k} runs on {args.rrt_maps} maps)\n"
        )

    header = f"{'checkpoint':<42}{'connected':>10}{'gap cells':>10}"
    header += f"{'walker':>8}{'walker@0':>9}{'divers.':>8}{'hedging':>9}"
    print(header)
    for ckpt in args.checkpoints:
        model = RidgeModel(ckpt, grid)
        conn, gaps, walk, walk0, div, hedge = [], [], [], [], [], []
        for begin in range(0, len(worlds), args.batch):
            chunk = worlds[begin : begin + args.batch]
            occ, grids, starts, goals = stack(chunk, model.device)
            si = torch.floor(starts / grid.cell).long()
            gi = torch.floor(goals / grid.cell).long()
            masks = []
            for j in range(args.k):
                h = model.sample(occ, starts, goals, seed=begin + 10_000 * j)
                conn += (bottleneck(h, ~grids, si, gi) >= ON_RIDGE).tolist()
                hedge += (hedging(h, ~grids, si, gi)[0] >= ON_RIDGE).tolist()
                route = geodesic_mask(h, ~grids, si, gi)
                gaps += (route & (h < ON_RIDGE)).sum((1, 2)).tolist()
                res = walker.walk(h, grids, starts, goals)
                ok = [
                    bool(r) and w.path_free(p)
                    for w, p, r in zip(chunk, res.paths, res.reached, strict=True)
                ]
                walk += ok
                if j == 0:
                    walk0 += ok
                masks.append((h >= ON_RIDGE).cpu().numpy())
            masks = np.stack(masks, 1)
            div += [diversity(m) for m in masks]
        print(
            f"{ckpt:<42}{100 * np.mean(conn):9.1f}%{np.mean(gaps):10.2f}"
            f"{100 * np.mean(walk):7.1f}%{100 * np.mean(walk0):8.1f}%{np.mean(div):8.3f}"
            f"{100 * np.mean(hedge):8.1f}%",
            flush=True,
        )
    print(
        "\nconnected: share of samples whose ridge (H >= 0.5) joins start and goal in free "
        "space, no corner cutting.\ngap cells: cells on the ridge geodesic with H < 0.5. "
        f"walker: over all {args.k} samples; walker@0: benchmark seed only.\n"
        "divers.: mean pairwise 1 - IoU of ridge masks across samples of one map.\n"
        "hedging: share of samples that also paint a second unbroken route outside a 3 px "
        "corridor around the main one."
    )


if __name__ == "__main__":
    main()
