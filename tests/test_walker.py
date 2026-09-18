from __future__ import annotations

import numpy as np

from ridgeflow import GridSpec, Rect, RidgeWalker, StepOutcome, WalkerConfig, World
from ridgeflow.targets import path_heatmap, world_to_pixel


def _straight_heatmap(grid: GridSpec, start, goal) -> np.ndarray:
    pixels = world_to_pixel(np.stack([start, goal]), grid)
    return np.ascontiguousarray(path_heatmap(pixels, grid.size, 1.5).numpy()[0].T)


def test_candidate_set_is_the_turn_speed_product(grid: GridSpec) -> None:
    start, goal = np.array([1.0, 4.0]), np.array([7.0, 4.0])
    config = WalkerConfig()
    walker = RidgeWalker(_straight_heatmap(grid, start, goal), start, goal, 0.5, grid, config)
    positions, headings, speeds = walker.candidates()
    assert len(positions) == config.turn_samples * len(config.speeds)
    assert len(headings) == len(positions) == len(speeds)
    assert set(np.unique(speeds)) == set(config.speeds)


def test_walker_is_pulled_back_onto_the_ridge(grid: GridSpec) -> None:
    """The objective supplies an across-ridge gradient; it is what recovers a drift."""
    ridge_start, ridge_goal = np.array([1.0, 4.0]), np.array([7.0, 4.0])
    heatmap = _straight_heatmap(grid, ridge_start, ridge_goal)
    offset = np.array([3.0, 5.5])
    world = World([], offset, ridge_goal, grid)
    walker = RidgeWalker(heatmap, offset, ridge_goal, 0.5, grid)
    before = abs(walker.position_world[1] - 4.0)
    for _ in range(40):
        walker.step(world.occupancy())
    assert abs(walker.position_world[1] - 4.0) < before


def test_a_static_field_gives_no_along_ridge_gradient(grid: GridSpec) -> None:
    """Every point on the ridge is equally close to it, so progress is not scored.

    In the simulation the direction comes from re-anchoring each replanned field ahead
    of the robot, not from this term. A never-updated field therefore cannot be used as
    an ablation baseline while goal_weight is zero.
    """
    start, goal = np.array([1.0, 4.0]), np.array([7.0, 4.0])
    world = World([], start, goal, grid)
    walker = RidgeWalker(_straight_heatmap(grid, start, goal), start, goal, 0.5, grid)
    for _ in range(400):
        if walker.step(world.occupancy()) is StepOutcome.ARRIVED:
            break
    assert walker.distance_to_goal > walker.config.goal_tolerance_px


def test_zero_speed_is_available_when_boxed_in(grid: GridSpec) -> None:
    start, goal = np.array([4.0, 4.0]), np.array([7.0, 4.0])
    walls = [
        Rect(4.3, 3.0, 0.5, 2.0, 0.0, 0.0),
        Rect(3.2, 3.0, 0.5, 2.0, 0.0, 0.0),
        Rect(3.2, 4.3, 1.6, 0.5, 0.0, 0.0),
        Rect(3.2, 3.0, 1.6, 0.5, 0.0, 0.0),
    ]
    world = World(walls, start, goal, grid)
    walker = RidgeWalker(_straight_heatmap(grid, start, goal), start, goal, 0.5, grid)
    assert walker.step(world.occupancy()) in {StepOutcome.WAITED, StepOutcome.MOVED}


def test_collision_filter_rejects_a_blocked_step(grid: GridSpec) -> None:
    start, goal = np.array([1.0, 4.0]), np.array([7.0, 4.0])
    world = World([Rect(1.05, 3.0, 2.0, 2.0, 0.0, 0.0)], start, goal, grid)
    walker = RidgeWalker(_straight_heatmap(grid, start, goal), start, goal, 0.5, grid)
    positions, _, _ = walker.candidates()
    keep = walker.feasible(positions, world.occupancy())
    assert not keep.all()


def test_score_is_scale_free(grid: GridSpec) -> None:
    start, goal = np.array([1.0, 4.0]), np.array([7.0, 4.0])
    heatmap = _straight_heatmap(grid, start, goal)
    walker = RidgeWalker(heatmap, start, goal, 0.5, grid)
    positions, _, speeds = walker.candidates()
    baseline = walker.score(positions, speeds)
    walker.set_heatmap(heatmap * 0.25)
    assert np.allclose(walker.score(positions, speeds), baseline, atol=1e-9)
