"""TMP64: Tiled Motion Planning queries on 64x64 grids, in the format ``RidgeDataset`` reads.

Following TransPath: each 64x64 map tiles four 32x32 maps drawn at random from all eight MP
categories (Neural A*'s ``planning-datasets``, ``*_032_moore_c8.npz``, 1 = free). Per map, a goal
is a random free cell and the start is drawn from the most distant third of the cells that can
reach it. The reference is an optimal path (A*, 8-connected, no corner cutting, octile costs).
No 64x64 map appears in two splits. Coordinates: cell (i, j) of the [x, y] grid has its centre at
world ((i + 0.5) * 0.125, (j + 0.5) * 0.125) on the default 64 x 8 m grid.
"""

from __future__ import annotations

import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ridgeflow.search import astar, distances_from

CELL = 0.125
SPLITS = {"train": 8140, "val": 1000, "test": 1044}


def load_tiles(folder: str) -> np.ndarray:
    tiles = []
    for f in sorted(glob.glob(f"{folder}/*_032_moore_c8.npz")):
        d = np.load(f)
        tiles += [d["arr_0"], d["arr_4"], d["arr_8"]]
    return np.concatenate(tiles) > 0.5  # True = free


def _queries(args):
    blocked, seed, per_map = args
    rng = np.random.default_rng(seed)
    free = np.argwhere(~blocked)
    out = []
    for _ in range(per_map * 4):
        if len(out) == per_map:
            break
        goal = tuple(free[rng.integers(len(free))])
        dist = distances_from(blocked, goal)
        reach = np.isfinite(dist) & (dist > 0)
        if reach.sum() < 64:
            continue
        far = reach & (dist >= np.quantile(dist[reach], 2.0 / 3.0))
        cells = np.argwhere(far)
        start = tuple(cells[rng.integers(len(cells))])
        res = astar(blocked, start, goal)
        out.append((start, goal, res.path, res.cost))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpd", default="outputs/data/mpd")
    parser.add_argument("--outdir", default="outputs/data")
    parser.add_argument("--per-map", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    tiles = load_tiles(args.mpd)
    rng = np.random.default_rng(args.seed)
    seen: set[bytes] = set()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for split, target in SPLITS.items():
        maps, rows = [], []
        while sum(len(r) for r in rows) < target:
            batch = []
            for _ in range(64):
                pick = tiles[rng.integers(len(tiles), size=4)]
                free = np.block([[pick[0], pick[1]], [pick[2], pick[3]]])
                key = free.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                batch.append(~free)
            with Pool(args.workers) as pool:
                found = pool.map(
                    _queries,
                    [(b, int(rng.integers(2**31)), args.per_map) for b in batch],
                )
            for blocked, qs in zip(batch, found, strict=True):
                if qs:
                    maps.append(blocked)
                    rows.append(qs)
        # trim to exactly the target number of queries
        flat = [(m, q) for m, qs in enumerate(rows) for q in qs][:target]
        used = sorted({m for m, _ in flat})
        remap = {m: k for k, m in enumerate(used)}
        data = {
            "map": np.stack([maps[m] for m in used]).astype(np.uint8),  # [x, y], 1 = blocked
            "map_id": np.array([remap[m] for m, _ in flat], np.int64),
            "start": np.array([(np.array(q[0]) + 0.5) * CELL for _, q in flat], np.float32),
            "goal": np.array([(np.array(q[1]) + 0.5) * CELL for _, q in flat], np.float32),
            "paths": [((q[2] + 0.5) * CELL).astype(np.float32) for _, q in flat],
            "start_cell": np.array([q[0] for _, q in flat], np.int64),
            "goal_cell": np.array([q[1] for _, q in flat], np.int64),
            "opt_cost": np.array([q[3] for _, q in flat], np.float64),
        }
        np.save(out / f"tmp64_{split}.npy", data, allow_pickle=True)
        occ = data["map"].mean()
        print(
            f"{split}: {len(flat)} queries on {len(used)} maps, obstacle {100 * occ:.1f}%, "
            f"mean optimal cost {data['opt_cost'].mean():.1f} cells",
            flush=True,
        )


if __name__ == "__main__":
    main()
