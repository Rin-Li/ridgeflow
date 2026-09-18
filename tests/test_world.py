from __future__ import annotations

import numpy as np
import pytest

from ridgeflow import GridSpec, Rect, World
from ridgeflow.scenarios import sample_pocket_world, sample_scattered_world


def test_rect_bounces_off_the_boundary() -> None:
    rect = Rect(7.5, 1.0, 1.0, 1.0, vx=1.0, vy=0.0)
    rect.advance(1.0, extent=8.0)
    assert rect.x == pytest.approx(7.0)
    assert rect.vx < 0.0


def test_rasterisation_covers_every_touched_cell(grid: GridSpec) -> None:
    rect = Rect(0.06, 0.06, 0.13, 0.13, 0.0, 0.0)
    world = World([rect], np.array([4.0, 4.0]), np.array([5.0, 5.0]), grid)
    occupancy = world.occupancy()
    assert occupancy[0, 0] == 1.0
    assert occupancy[1, 1] == 1.0
    assert occupancy.sum() == 4.0


def test_collision_agrees_with_the_raster(grid: GridSpec) -> None:
    rect = Rect(2.0, 2.0, 1.0, 1.0, 0.0, 0.0)
    world = World([rect], np.array([0.5, 0.5]), np.array([7.0, 7.0]), grid)
    occupied = np.argwhere(world.occupancy() > 0.5)
    for index in occupied:
        centre = (index + 0.5) * grid.cell
        assert world.blocked(centre)


def test_outside_the_grid_counts_as_blocked(grid: GridSpec) -> None:
    world = World([], np.array([1.0, 1.0]), np.array([2.0, 2.0]), grid)
    assert world.blocked([-0.1, 1.0])
    assert world.blocked([8.1, 1.0])
    assert not world.blocked([4.0, 4.0])


def test_boxes_steer_away_from_the_robot(grid: GridSpec) -> None:
    rect = Rect(3.0, 3.9, 0.4, 0.4, vx=1.0, vy=0.0)
    world = World([rect], np.array([0.5, 0.5]), np.array([7.0, 7.0]), grid)
    speed_before = rect.speed
    world.advance(0.05, robot=np.array([3.6, 4.1]), avoid_radius=1.5, avoid_gain=2.0)
    assert rect.speed == pytest.approx(speed_before)
    assert rect.vx < 1.0


def test_goal_bay_is_never_sealed(grid: GridSpec) -> None:
    goal = np.array([4.0, 4.0])
    rect = Rect(3.8, 3.8, 0.4, 0.4, vx=0.0, vy=0.0)
    world = World([rect], np.array([0.5, 0.5]), goal, grid)
    world.advance(0.05, keep_clear=(goal,), bay_half=0.45)
    assert not world.blocked(goal)


def test_pocket_layout_is_connected_and_blocks_the_direct_line(grid: GridSpec) -> None:
    world = sample_pocket_world(np.random.default_rng(3), grid, speed=1.0)
    assert not world.blocked(world.start)
    assert not world.blocked(world.goal)
    assert world.segment_blocked(world.start, world.goal)
    assert 0.10 <= world.occupancy().mean() <= 0.34


def test_scattered_layout_respects_its_density_band(grid: GridSpec) -> None:
    world = sample_scattered_world(np.random.default_rng(1), grid, density="medium", speed=1.0)
    assert 0.10 <= world.occupancy().mean() <= 0.30
    assert world.segment_blocked(world.start, world.goal)
