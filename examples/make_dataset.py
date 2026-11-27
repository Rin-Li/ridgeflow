"""Generate RRT* demonstrations on random maps, in the format ``RidgeDataset`` reads.

The planner (``ridgeflow.planners.RRTStar``) is the one the shipped checkpoint was trained
with: the first solution of RRT*, then shortcut-pruned. One query per map; seeds are disjoint from
benchmark.py (0..999).
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ridgeflow import GridSpec, sample_rect_world
from ridgeflow.planners import RRTStar


def _one(seed: int):
    import torch

    torch.set_num_threads(1)
    grid = GridSpec()
    world = sample_rect_world(np.random.default_rng(seed), grid)
    path = RRTStar(world, rng=seed).plan()
    if path is None:
        return None
    return world.occupancy().numpy(), world.start, world.goal, path


def build(seeds, workers: int) -> dict:
    with Pool(workers) as pool:
        rows = [r for r in pool.imap(_one, seeds, chunksize=16) if r is not None]
    return {
        "map": np.stack([r[0] for r in rows]).astype(np.uint8),
        "start": np.stack([r[1] for r in rows]).astype(np.float32),
        "goal": np.stack([r[2] for r in rows]).astype(np.float32),
        "paths": [r[3].astype(np.float32) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=8140)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--outdir", default="outputs/data")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, base, count in (("train", 100_000, args.train), ("val", 200_000, args.val)):
        data = build(range(base, base + count), args.workers)
        np.save(out / f"rrt64_{name}.npy", data, allow_pickle=True)
        print(f"{name}: {len(data['paths'])}/{count} solved -> {out / f'rrt64_{name}.npy'}")


if __name__ == "__main__":
    main()
