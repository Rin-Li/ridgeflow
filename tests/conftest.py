from __future__ import annotations

import numpy as np
import pytest
import torch

from ridgeflow import GridSpec
from ridgeflow.model.targets import path_heatmap, world_to_pixel


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(size=64, extent=8.0)


def ridge(grid: GridSpec, *points) -> torch.Tensor:
    """A perfect training-style ridge through ``points``, as a ``(1, S, S)`` [x, y] tensor."""
    pixels = world_to_pixel(np.stack(points), grid)
    return path_heatmap(pixels, grid.size).transpose(1, 2).contiguous()
