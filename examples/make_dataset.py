"""Generate RRT* demonstrations on random maps, in the format ``RidgeDataset`` reads.

The planner is the one the shipped checkpoint was trained with (restored from 148f6a3^):
first solution of RRT*, then shortcut-pruned. One query per map; seeds are disjoint from
benchmark.py (0..999).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from math import log
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ridgeflow import GridSpec, World, sample_rect_world


@dataclass(frozen=True)
class RRTStarConfig:
    max_iter: int = 3000
    step_size: float = 0.5
    goal_tol: float = 0.3
    goal_bias: float = 0.1
    gamma_star: float = 1.5
    prune: bool = True


class RRTStar:
    """RRT* from the training-data generator: first solution, then shortcut-pruned."""

    def __init__(self, world: World, config: RRTStarConfig | None = None, rng=None) -> None:
        self.world = world
        self.config = config or RRTStarConfig()
        self.rng = np.random.default_rng(rng)

    def _sample_free(self) -> np.ndarray:
        for _ in range(1000):
            point = self.rng.uniform(0.0, self.world.grid.extent, size=2)
            if not self.world.blocked(point):
                return point
        raise RuntimeError("failed to sample a free point")

    def _steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        offset = target - source
        distance = float(np.linalg.norm(offset))
        if distance <= self.config.step_size:
            return target.copy()
        return source + offset / distance * self.config.step_size

    def _search(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        world, config = self.world, self.config
        capacity = config.max_iter + 1
        points = np.empty((capacity, 2))
        parents = np.full(capacity, -1, int)
        costs = np.zeros(capacity)
        points[0] = start
        count = 1
        for iteration in range(1, config.max_iter + 1):
            target = goal.copy() if self.rng.random() < config.goal_bias else self._sample_free()
            nearest = int(np.argmin(np.linalg.norm(points[:count] - target, axis=1)))
            new = self._steer(points[nearest], target)
            if world.segment_blocked(points[nearest], new):
                continue
            radius = min(
                config.gamma_star * (log(iteration) / iteration) ** 0.5, 2 * config.step_size
            )
            distances = np.linalg.norm(points[:count] - new, axis=1)
            neighbours = [
                int(i)
                for i in np.flatnonzero(distances <= radius)
                if not world.segment_blocked(points[i], new)
            ]
            parent = min(neighbours or [nearest], key=lambda i: costs[i] + distances[i])
            index = count
            points[index], parents[index] = new, parent
            costs[index] = costs[parent] + distances[parent]
            count += 1
            for i in neighbours:
                through = costs[index] + distances[i]
                if through < costs[i]:
                    parents[i], costs[i] = index, through
            if np.linalg.norm(new - goal) <= config.goal_tol and not world.segment_blocked(
                new, goal
            ):
                path, node = [goal], index
                while node >= 0:
                    path.append(points[node].copy())
                    node = parents[node]
                return np.asarray(path[::-1])
        return None

    def _prune(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path
        pruned, anchor = [path[0]], path[0]
        for i in range(2, len(path)):
            if self.world.segment_blocked(anchor, path[i]):
                pruned.append(path[i - 1])
                anchor = path[i - 1]
        pruned.append(path[-1])
        return np.asarray(pruned)

    def plan(self) -> np.ndarray | None:
        start = np.asarray(self.world.start, np.float64)
        goal = np.asarray(self.world.goal, np.float64)
        raw = self._search(start, goal)
        if raw is None:
            return None
        path = self._prune(raw) if self.config.prune else raw
        if not self.world.path_free(path):
            path = raw
        return path if self.world.path_free(path) else None


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
