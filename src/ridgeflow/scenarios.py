from __future__ import annotations

import numpy as np
from scipy import ndimage

from ridgeflow.grid import GridSpec
from ridgeflow.world import Rect, World, any_overlap

DENSITY_BANDS = {
    "sparse": ((3, 5), (0.04, 0.10)),
    "medium": ((6, 10), (0.10, 0.30)),
    "dense": ((12, 20), (0.30, 0.42)),
}

EIGHT_CONNECTED = np.ones((3, 3), bool)


def _drifting_rect(rng, x: float, y: float, width: float, height: float, speed: float) -> Rect:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return Rect(x, y, width, height, speed * np.cos(angle), speed * np.sin(angle))


def _connected(world: World, a, b) -> bool:
    labels, _ = ndimage.label(world.occupancy() <= 0.5, structure=EIGHT_CONNECTED)
    ia, ib = world.grid.index(a), world.grid.index(b)
    if not (world.grid.holds(ia) and world.grid.holds(ib)):
        return False
    return labels[ia] > 0 and labels[ia] == labels[ib]


def sample_scattered_world(
    rng,
    grid: GridSpec | None = None,
    density: str = "medium",
    speed: float = 1.0,
    min_span: float = 4.0,
    tries: int = 400,
) -> World:
    """Free-floating boxes at the density the generative model was trained on.

    The straight line from start to goal is required to hit something, which is the
    training generator's own criterion; without it most queries are solved by walking
    at the goal and guidance has nothing to contribute.
    """
    grid = grid or GridSpec()
    count_band, occupancy_band = DENSITY_BANDS[density]
    extent = grid.extent
    for _ in range(tries):
        rects: list[Rect] = []
        for _ in range(int(rng.integers(count_band[0], count_band[1] + 1))):
            for _ in range(100):
                width, height = rng.uniform(0.5, 2.4, size=2)
                candidate = _drifting_rect(
                    rng,
                    rng.uniform(0.0, extent - width),
                    rng.uniform(0.0, extent - height),
                    width,
                    height,
                    speed,
                )
                if not any_overlap(candidate, rects):
                    rects.append(candidate)
                    break
        world = World(rects, np.zeros(2), np.zeros(2), grid)
        if not occupancy_band[0] <= float(world.occupancy().mean()) <= occupancy_band[1]:
            continue
        for _ in range(200):
            start = rng.uniform(0.3, extent - 0.3, size=2)
            goal = rng.uniform(0.3, extent - 0.3, size=2)
            if np.linalg.norm(goal - start) < min_span:
                continue
            if world.blocked(start) or world.blocked(goal):
                continue
            if not world.segment_blocked(start, goal):
                continue
            world.start, world.goal = start, goal
            return world
    raise RuntimeError("could not lay out a scattered world")


def _span_box(a, b, thickness: float, extent: float) -> Rect:
    low = np.clip(np.minimum(a, b) - thickness / 2.0, 0.0, extent)
    high = np.clip(np.maximum(a, b) + thickness / 2.0, 0.0, extent)
    return Rect(
        float(low[0]),
        float(low[1]),
        float(max(high[0] - low[0], 0.2)),
        float(max(high[1] - low[1], 0.2)),
    )


def _pocket_walls(
    centre, mouth_direction, extent: float, arm: float = 1.9, width: float = 1.5,
    thickness: float = 0.42,
) -> list[Rect]:
    """Three static boxes forming a U whose opening faces ``mouth_direction``."""
    forward = np.asarray(mouth_direction, np.float64)
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    normal = np.array([-forward[1], forward[0]])
    back = np.asarray(centre, np.float64) - forward * arm / 2.0
    walls = [_span_box(back - normal * width / 2.0, back + normal * width / 2.0, thickness, extent)]
    for side in (-1.0, 1.0):
        anchor = back + normal * side * width / 2.0
        walls.append(_span_box(anchor, anchor + forward * arm, thickness, extent))
    return walls


def sample_pocket_world(
    rng,
    grid: GridSpec | None = None,
    movers: int = 4,
    speed: float = 1.0,
    min_span: float = 5.0,
    tries: int = 300,
) -> World:
    """A U-shaped dead end straddling the direct line, plus drifting boxes.

    Walking at the goal enters the pocket and has to reverse out of it, so the layout
    separates a controller that follows a global route from one that follows the goal.
    """
    grid = grid or GridSpec()
    extent = grid.extent
    for _ in range(tries):
        start = rng.uniform(0.4, extent - 0.4, size=2)
        goal = rng.uniform(0.4, extent - 0.4, size=2)
        span = float(np.linalg.norm(goal - start))
        if span < min_span:
            continue
        direction = (goal - start) / span
        centre = start + direction * span * rng.uniform(0.45, 0.62)
        walls = _pocket_walls(centre, -direction, extent)

        boxes: list[Rect] = []
        for _ in range(movers):
            for _ in range(80):
                width, height = rng.uniform(0.5, 1.4, size=2)
                candidate = _drifting_rect(
                    rng,
                    rng.uniform(0.0, extent - width),
                    rng.uniform(0.0, extent - height),
                    width,
                    height,
                    speed,
                )
                if not any_overlap(candidate, walls + boxes, pad=0.15):
                    boxes.append(candidate)
                    break

        world = World(walls + boxes, start, goal, grid)
        if world.blocked(start) or world.blocked(goal):
            continue
        if not _connected(world, start, goal):
            continue
        if not 0.10 <= float(world.occupancy().mean()) <= 0.34:
            continue
        return world
    raise RuntimeError("could not lay out a pocket world")
