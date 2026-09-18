from __future__ import annotations

import numpy as np
import pytest

from ridgeflow import GridSpec
from ridgeflow.targets import path_heatmap, world_to_pixel


class StraightLineGuidance:
    """A Guidance that renders a perfect ridge along the straight start-goal line."""

    def __init__(self, grid: GridSpec, sigma: float = 1.5) -> None:
        self.grid = grid
        self.sigma = sigma
        self.calls = 0

    def heatmap(self, occupancy, start_world, goal_world, seed: int = 0) -> np.ndarray:
        self.calls += 1
        path = np.stack([np.asarray(start_world), np.asarray(goal_world)])
        pixels = world_to_pixel(path, self.grid)
        raster = path_heatmap(pixels, self.grid.size, self.sigma).numpy()[0]
        return np.ascontiguousarray(raster.T)

    @property
    def ms_per_call(self) -> float:
        return 0.0


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(size=64, extent=8.0)


@pytest.fixture
def guidance(grid: GridSpec) -> StraightLineGuidance:
    return StraightLineGuidance(grid)
