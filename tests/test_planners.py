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
        config.amplitude * np.exp(-((k * config.step_px) ** 2) / (2 * config.sigma**2))
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


def test_boxed_in_walker_stops_when_it_may_not_reverse(grid: GridSpec) -> None:
    walls = [Rect(4.1, 3.0, 0.5, 2.0), Rect(3.2, 3.3, 1.6, 0.5), Rect(3.2, 4.1, 1.6, 0.5)]
    world = World(walls, np.array([3.95, 3.95]), np.array([7.0, 4.0]), grid)
    result = _walk(
        world, ridge(grid, world.start, world.goal), reverse_when_stuck=False, repair=False
    )
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


def _px(*pixels):
    """Pixel centres -> world points on the default 64 x 8 grid."""
    return [np.array([(x + 0.5) * 0.125, (y + 0.5) * 0.125]) for x, y in pixels]


def _dead_end(grid: GridSpec):
    """A one-cell corridor (row 32, x 20..39) capped at x = 40; the start sits inside it and the
    goal lies straight ahead beyond the cap. Ridge A runs straight at the goal (a dead end);
    ridge B leaves the corridor backwards and goes round."""
    grids = torch.zeros(1, grid.size, grid.size, dtype=torch.bool)
    grids[0, 20:41, 31] = True
    grids[0, 20:41, 33] = True
    grids[0, 40, 31:34] = True
    start, goal = _px((35, 32), (50, 32))
    a = ridge(grid, start, goal)
    b = ridge(grid, start, *_px((10, 32), (10, 50), (50, 50)), goal)
    starts = torch.tensor(start[None], dtype=torch.float32)
    goals = torch.tensor(goal[None], dtype=torch.float32)
    return torch.maximum(a, b), grids, starts, goals


@pytest.mark.parametrize("backend", ["torch", "numba"])
def test_walker_reverses_out_of_a_dead_end(grid: GridSpec, backend: str) -> None:
    heatmap, grids, starts, goals = _dead_end(grid)
    stuck = RidgeWalker(
        grid,
        WalkerConfig(backend=backend, reverse_when_stuck=False, repair=False, backtrack=False),
    )
    assert not stuck.walk(heatmap, grids, starts, goals).reached[0]
    walker = RidgeWalker(grid, WalkerConfig(backend=backend, repair=False, backtrack=False))
    result = walker.walk(heatmap, grids, starts, goals)
    assert result.reached[0]
    assert result.paths[0][:, 1].max() > 6.0  # it went round, over the corridor


def test_backends_walk_the_same_path(grid: GridSpec) -> None:
    worlds = [
        World([], np.array([1.0, 1.0]), np.array([7.0, 1.0]), grid),
        World([Rect(3.5, 2.0, 1.0, 4.0)], np.array([1.0, 1.0]), np.array([7.0, 7.0]), grid),
    ]
    heatmaps = torch.cat(
        [
            ridge(grid, worlds[0].start, [1.0, 6.0], [7.0, 6.0], worlds[0].goal),
            ridge(grid, worlds[1].start, [2.0, 7.0], worlds[1].goal),
        ]
    )
    _, grids, starts, goals = stack(worlds, "cpu")
    paths = {
        b: RidgeWalker(grid, WalkerConfig(backend=b, backtrack=False)).walk(
            heatmaps, grids, starts, goals
        )
        for b in ("torch", "numba")
    }
    for a, b in zip(paths["torch"].paths, paths["numba"].paths, strict=True):
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-4)
    assert (paths["torch"].reached == paths["numba"].reached).all()


def test_walker_connects_to_a_goal_within_one_ridge_width(grid: GridSpec) -> None:
    assert WalkerConfig().arrival_radius == pytest.approx(2 * WalkerConfig().sigma)
    assert WalkerConfig(arrive_px=1.0).arrival_radius == 1.0
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    result = _walk(world, ridge(grid, world.start, [7.0, 4.25]))  # ends 2 px beside the goal
    assert result.reached[0]
    assert np.allclose(result.paths[0][-1], world.goal)


def test_route_corridor_hides_ridge_away_from_the_route(grid: GridSpec) -> None:
    heatmap, grids, starts, goals = _dead_end(grid)
    route = torch.zeros_like(grids)
    route[0, 55:60, 5:10] = True  # a corner the model drew nothing in
    walker = RidgeWalker(grid, WalkerConfig(repair=False))
    result = walker.walk(heatmap, grids, starts, goals, route=route)
    assert float(result.residual.max()) < 1e-6
    assert not result.reached[0]


def _free(grids: torch.Tensor, path: np.ndarray, grid: GridSpec) -> bool:
    from ridgeflow.collision import segments_blocked

    a = torch.as_tensor(path[:-1], dtype=torch.float32)
    b = torch.as_tensor(path[1:], dtype=torch.float32)[:, None]
    return not bool(segments_blocked(grids.expand(len(a), -1, -1), a, b, grid.cell).any())


@pytest.mark.parametrize("backend", ["torch", "numba"])
def test_repair_finishes_a_walk_that_got_stuck(grid: GridSpec, backend: str) -> None:
    heatmap, grids, starts, goals = _dead_end(grid)
    config = WalkerConfig(backend=backend, reverse_when_stuck=False, backtrack=False, repair=True)
    result = RidgeWalker(grid, config).walk(heatmap, grids, starts, goals)
    path = result.paths[0]
    assert result.reached[0] and result.repaired[0]
    assert np.allclose(path[0], starts[0]) and np.allclose(path[-1], goals[0])
    assert _free(grids, path, grid)


def test_repair_needs_no_ridge_at_all(grid: GridSpec) -> None:
    _, grids, starts, goals = _dead_end(grid)
    empty = torch.zeros(1, grid.size, grid.size)
    result = RidgeWalker(grid, WalkerConfig(repair=True)).walk(empty, grids, starts, goals)
    assert result.reached[0] and result.repaired[0]
    assert _free(grids, result.paths[0], grid)
    # string pulling leaves few vertices: no grid staircase
    assert len(result.paths[0]) < 15


def test_repair_does_not_invent_a_way_to_an_unreachable_goal(grid: GridSpec) -> None:
    heatmap, grids, starts, goals = _dead_end(grid)
    grids = grids.clone()
    grids[0, 45:56, 27:38] = True
    grids[0, 47:54, 29:36] = False  # a sealed room around the goal
    grids[0, 50, 32] = False
    result = RidgeWalker(grid, WalkerConfig(repair=True)).walk(heatmap, grids, starts, goals)
    assert not result.reached[0] and not result.repaired[0]


def _fork(grid: GridSpec):
    """A wall at x = 40 from y = 10 up; open below. The bright ridge runs straight at the goal
    into the wall (a dead end); at x = 25 a fainter branch turns off, under the wall's end."""
    grids = torch.zeros(1, grid.size, grid.size, dtype=torch.bool)
    grids[0, 39:42, 10:] = True
    start, goal = _px((10, 32), (54, 32))
    bait = ridge(grid, start, goal)
    branch = 0.7 * ridge(grid, *_px((25, 32), (34, 4), (46, 4)), goal)
    starts = torch.tensor(start[None], dtype=torch.float32)
    goals = torch.tensor(goal[None], dtype=torch.float32)
    return torch.maximum(bait, branch), grids, starts, goals


def test_backtracking_walker_drops_the_dead_end_it_tried(grid: GridSpec) -> None:
    heatmap, grids, starts, goals = _fork(grid)

    def near_wall(path):  # points above the wall's lower end, right next to it
        return int(((path[:, 0] > 4.4) & (path[:, 0] < 5.0) & (path[:, 1] > 1.6)).sum())

    greedy = RidgeWalker(grid, WalkerConfig(backtrack=False)).walk(heatmap, grids, starts, goals)
    result = RidgeWalker(grid).walk(heatmap, grids, starts, goals)
    path = result.paths[0]
    assert result.reached[0] and result.backtracks[0] >= 1
    assert _free(grids, path, grid)
    # the greedy walker keeps its whole detour along the dead end; the backtracking walker
    # drops what it walked after the latest branch it returned to
    assert near_wall(path) < near_wall(greedy.paths[0]) / 4
    assert len(path) < len(greedy.paths[0])
