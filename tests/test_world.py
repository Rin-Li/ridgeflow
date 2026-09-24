from __future__ import annotations

import numpy as np
import torch

from ridgeflow import SCENES, GridSpec, Rect, World, sample_rect_world
from ridgeflow.collision import blocked
from ridgeflow.scenarios import connected


def test_rasterisation_uses_cell_centres(grid: GridSpec) -> None:
    world = World([Rect(0.0, 0.0, 0.1, 0.1)], np.array([4.0, 4.0]), np.array([5.0, 5.0]), grid)
    assert world.occupancy().sum() == 1
    assert world.occupancy()[0, 0]


def test_planning_grid_is_padded_by_one_cell(grid: GridSpec) -> None:
    world = World([Rect(4.0, 4.0, 0.1, 0.1)], np.array([1.0, 1.0]), np.array([7.0, 7.0]), grid)
    assert world.planning_grid().sum() == 9


def test_outside_the_grid_is_blocked(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 1.0]), np.array([2.0, 2.0]), grid)
    assert world.blocked([-0.1, 1.0])
    assert world.blocked([8.1, 1.0])
    assert not world.blocked([4.0, 4.0])


def test_probe_catches_a_wall_just_across_the_cell_edge(grid: GridSpec) -> None:
    raster = torch.zeros(1, grid.size, grid.size, dtype=torch.bool)
    raster[0, 10, 10] = True
    edge = 10 * grid.cell
    near = torch.tensor([[[edge - 0.05 * grid.cell, 10.5 * grid.cell]]])
    far = torch.tensor([[[edge - 0.5 * grid.cell, 10.5 * grid.cell]]])
    assert blocked(raster, near, grid.cell).item()
    assert not blocked(raster, far, grid.cell).item()


def test_path_free_checks_every_segment(grid: GridSpec) -> None:
    world = World([Rect(3.5, 0.0, 1.0, 8.0)], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    assert not world.path_free([world.start, world.goal])
    assert world.path_free([[1.0, 1.0], [3.0, 1.0]])
    assert not connected(world)


def test_random_worlds_are_solvable_and_blocked(grid: GridSpec) -> None:
    for seed in range(5):
        world = sample_rect_world(np.random.default_rng(seed), grid)
        assert len(world.rects) <= 10
        assert not world.blocked(world.start) and not world.blocked(world.goal)
        assert world.segment_blocked(world.start, world.goal)
        assert connected(world)


def test_scenes_are_solvable_and_blocked(grid: GridSpec) -> None:
    for make in SCENES.values():
        world = make(grid)
        assert not world.blocked(world.start) and not world.blocked(world.goal)
        assert world.segment_blocked(world.start, world.goal)
        assert connected(world)
