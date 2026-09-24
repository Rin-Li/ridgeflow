from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from ridgeflow.world import World


@dataclass(frozen=True)
class RRTStarConfig:
    """The settings the training demonstrations were generated with."""

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
        self.iterations = 0

    def _sample_free(self) -> np.ndarray:
        extent = self.world.grid.extent
        for _ in range(1000):
            point = self.rng.uniform(0.0, extent, size=2)
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
            self.iterations = iteration
            target = goal.copy() if self.rng.random() < config.goal_bias else self._sample_free()
            nearest = int(np.argmin(np.linalg.norm(points[:count] - target, axis=1)))
            new = self._steer(points[nearest], target)
            if world.segment_blocked(points[nearest], new):
                continue

            shrinking = config.gamma_star * (log(iteration) / iteration) ** 0.5
            radius = min(shrinking, 2 * config.step_size)
            distances = np.linalg.norm(points[:count] - new, axis=1)
            neighbours = [
                int(i)
                for i in np.flatnonzero(distances <= radius)
                if not world.segment_blocked(points[i], new)
            ]
            candidates = neighbours or [nearest]
            parent = min(candidates, key=lambda i: costs[i] + distances[i])

            index = count
            points[index] = new
            parents[index] = parent
            costs[index] = costs[parent] + distances[parent]
            count += 1

            for i in neighbours:
                through = costs[index] + distances[i]
                if through < costs[i]:
                    parents[i], costs[i] = index, through

            if np.linalg.norm(new - goal) <= config.goal_tol and not world.segment_blocked(
                new, goal
            ):
                path = [goal]
                node = index
                while node >= 0:
                    path.append(points[node].copy())
                    node = parents[node]
                return np.asarray(path[::-1])
        return None

    def _prune(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path
        pruned = [path[0]]
        anchor = path[0]
        for i in range(2, len(path)):
            if self.world.segment_blocked(anchor, path[i]):
                pruned.append(path[i - 1])
                anchor = path[i - 1]
        pruned.append(path[-1])
        return np.asarray(pruned)

    def plan(self) -> np.ndarray | None:
        start = np.asarray(self.world.start, np.float64)
        goal = np.asarray(self.world.goal, np.float64)
        if self.world.blocked(start) or self.world.blocked(goal):
            raise ValueError("start or goal is blocked")

        raw = self._search(start, goal)
        if raw is None:
            return None
        path = raw
        if self.config.prune:
            path = self._prune(raw)
            if not self.world.path_free(path):
                path = raw
        return path if self.world.path_free(path) else None
