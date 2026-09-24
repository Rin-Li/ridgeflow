from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import ridge

from ridgeflow import DWA, SCENES, GridSpec, Rect, RidgeWalker, WalkerConfig, World, stack
from ridgeflow.planners.dwa import clearance_field


def _walk(world: World, heatmap: torch.Tensor, **config):
    _, grids, starts, goals = stack([world], "cpu")
    return RidgeWalker(world.grid, WalkerConfig(**config)).walk(heatmap, grids, starts, goals)


def test_walker_constants_follow_from_the_ridge_and_the_grid() -> None:
    config = WalkerConfig()
    assert config.sigma == 1.5
    assert config.window_radius == 7
    line = sum(
        config.amplitude * np.exp(-(k * config.step_px) ** 2 / (2 * config.sigma**2))
        for k in range(-20, 21)
    )
    assert line == pytest.approx(1.0, rel=1e-3)


def test_walker_follows_a_straight_ridge_to_the_goal(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    result = _walk(world, ridge(grid, world.start, world.goal))
    path = result.paths[0]
    assert result.reached[0]
    assert np.allclose(path[-1], world.goal, atol=1e-5)
    assert np.abs(path[:, 1] - 4.0).max() < 0.1


def test_residual_alone_steers_around_a_bend(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 1.0]), np.array([7.0, 1.0]), grid)
    result = _walk(world, ridge(grid, world.start, [1.0, 6.0], [7.0, 6.0], world.goal))
    assert result.reached[0]
    assert result.paths[0][:, 1].max() > 5.0


def test_walking_consumes_the_ridge(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    heatmap = ridge(grid, world.start, world.goal)
    result = _walk(world, heatmap)
    assert result.residual.sum() < 0.2 * heatmap.sum()
    assert (result.residual >= 0).all()


def test_boxed_in_walker_stops(grid: GridSpec) -> None:
    walls = [Rect(4.1, 3.0, 0.5, 2.0), Rect(3.2, 3.3, 1.6, 0.5), Rect(3.2, 4.1, 1.6, 0.5)]
    world = World(walls, np.array([3.95, 3.95]), np.array([7.0, 4.0]), grid)
    result = _walk(world, ridge(grid, world.start, world.goal))
    assert not result.reached[0]
    assert len(result.paths[0]) == 1


def test_queries_in_a_batch_do_not_interact(grid: GridSpec) -> None:
    a = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    b = World([Rect(3.5, 2.0, 1.0, 4.0)], np.array([1.0, 1.0]), np.array([7.0, 7.0]), grid)
    heatmaps = torch.cat([ridge(grid, a.start, a.goal), ridge(grid, b.start, [2.0, 7.0], b.goal)])
    _, grids, starts, goals = stack([a, b], "cpu")
    together = RidgeWalker(grid).walk(heatmaps, grids, starts, goals)
    alone = _walk(b, heatmaps[1:])
    assert np.allclose(together.paths[1], alone.paths[0])


def test_dwa_reaches_an_open_goal(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    _, grids, starts, goals = stack([world], "cpu")
    assert DWA(grid).run(grids, starts, goals).reached[0]


def test_dwa_is_caught_by_the_u_trap(grid: GridSpec) -> None:
    _, grids, starts, goals = stack([SCENES["U-trap"](grid)], "cpu")
    assert not DWA(grid).run(grids, starts, goals).reached[0]


def test_clearance_is_zero_on_walls_and_grows_away(grid: GridSpec) -> None:
    world = World([Rect(3.5, 0.0, 1.0, 8.0)], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    field = clearance_field(world.planning_grid()[None], grid)[0]
    assert field[32, 32] == 0.0
    assert field[10, 32] > field[20, 32] > 0.0

