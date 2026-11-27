from __future__ import annotations

import numpy as np
import torch

from ridgeflow.model.targets import path_heatmap
from ridgeflow.routes import ON_RIDGE, bottleneck, hedging, project

S = 64


def _maps():
    """A wall in the middle; start left of it, goal right of it ([row, col] = [x, y])."""
    free = torch.ones(1, S, S, dtype=torch.bool)
    free[0, 26:38, 20:44] = False
    start, goal = torch.tensor([[32, 10]]), torch.tensor([[32, 54]])
    return free, start, goal


def _ridge(*points) -> torch.Tensor:
    # path_heatmap is [y, x] for [x, y] points; with points given as [col, row] it is [row, col]
    return path_heatmap(np.array(points, np.float32), S)


def test_bottleneck_and_projection_on_a_connected_ridge() -> None:
    free, start, goal = _maps()
    h = _ridge([10, 32], [10, 15], [54, 15], [54, 32])
    assert float(bottleneck(h, free, start, goal)) > 0.9
    projected, route = project(h, free, start, goal)
    assert torch.equal(projected, torch.where(route, h.clamp_min(ON_RIDGE), h))
    assert torch.allclose(projected, h)  # already connected: nothing to lift
    assert bool(route[0, 32, 10]) and bool(route[0, 32, 54])


def test_projection_closes_a_gap_and_only_touches_its_route() -> None:
    free, start, goal = _maps()
    h = torch.maximum(_ridge([10, 32], [10, 15], [27, 15]), _ridge([37, 15], [54, 15], [54, 32]))
    assert float(bottleneck(h, free, start, goal)) < ON_RIDGE
    projected, route = project(h, free, start, goal)
    assert float(bottleneck(projected, free, start, goal)) >= ON_RIDGE
    changed = projected != h
    assert bool((changed <= route).all())
    assert int(changed.sum()) <= 12  # just the gap, not a new route
    assert torch.allclose(projected[changed], torch.full_like(projected[changed], ON_RIDGE))


def test_hedging_sees_a_second_route_but_not_a_single_one() -> None:
    free, start, goal = _maps()
    one = _ridge([10, 32], [10, 15], [54, 15], [54, 32])
    two = torch.maximum(one, _ridge([10, 32], [10, 50], [54, 50], [54, 32]))
    b2, _ = hedging(
        torch.cat([one, two]), free.expand(2, S, S), start.repeat(2, 1), goal.repeat(2, 1)
    )
    assert float(b2[0]) < ON_RIDGE <= float(b2[1])
