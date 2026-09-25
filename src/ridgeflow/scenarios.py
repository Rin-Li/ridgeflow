from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ridgeflow.grid import GridSpec
from ridgeflow.world import Rect, World


def connected(world: World) -> bool:
    """Eight-connected flood fill from start to goal over the free planning cells."""
    free = ~world.planning_grid()
    start, goal = world.grid.index(world.start), world.grid.index(world.goal)
    reach = torch.zeros_like(free)
    reach[start] = True
    reach &= free
    while True:
        grown = (F.max_pool2d(reach.float()[None, None], 3, 1, 1)[0, 0] > 0.5) & free
        if bool(grown[goal]):
            return True
        if torch.equal(grown, reach):
            return False
        reach = grown


def _random_rects(rng, count: tuple[int, int], extent: float) -> list[Rect]:
    rects: list[Rect] = []
    for _ in range(int(rng.integers(count[0], count[1] + 1))):
        for _ in range(100):
            w, h = rng.uniform(0.5, 0.3 * extent, size=2)
            rect = Rect(rng.uniform(0.0, extent - w), rng.uniform(0.0, extent - h), w, h)
            if not any(rect.overlaps(other) for other in rects):
                rects.append(rect)
                break
    return rects


def sample_rect_world(
    rng, grid: GridSpec | None = None, rect_count: tuple[int, int] = (7, 10)
) -> World:
    """A query from the training distribution: 7-10 boxes, direct line blocked, solvable."""
    grid = grid or GridSpec()
    while True:
        world = World(_random_rects(rng, rect_count, grid.extent), np.zeros(2), np.zeros(2), grid)
        for _ in range(1000):
            start, goal = rng.uniform(0.0, grid.extent, size=(2, 2))
            if np.linalg.norm(goal - start) >= 1.0 and not (
                world.blocked(start) or world.blocked(goal)
            ):
                break
        else:
            continue
        world.start, world.goal = start, goal
        if world.segment_blocked(start, goal) and connected(world):
            return world


def _box(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return Rect(x0, y0, x1 - x0, y1 - y0)


def u_trap(grid: GridSpec | None = None) -> World:
    rects = [_box(4.6, 2.2, 5.1, 5.8), _box(2.6, 2.2, 4.6, 2.7), _box(2.6, 5.3, 4.6, 5.8)]
    return World(rects, np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid or GridSpec())


def comb(grid: GridSpec | None = None) -> World:
    rects = [
        _box(4.8, 1.2, 5.3, 6.8), _box(3.0, 1.2, 4.8, 1.7),
        _box(3.0, 3.75, 4.8, 4.25), _box(3.0, 6.3, 4.8, 6.8),
    ]
    return World(rects, np.array([1.2, 3.0]), np.array([7.0, 3.0]), grid or GridSpec())


def dead_end(grid: GridSpec | None = None) -> World:
    rects = [
        _box(2.8, 4.8, 5.2, 5.3), _box(2.8, 2.6, 3.3, 4.8), _box(4.7, 2.6, 5.2, 4.8),
        _box(1.0, 1.0, 2.2, 2.0), _box(6.0, 6.0, 7.2, 7.2),
    ]
    return World(rects, np.array([4.0, 3.4]), np.array([4.0, 7.0]), grid or GridSpec())


SCENES = {"U-trap": u_trap, "comb": comb, "dead-end start": dead_end}
