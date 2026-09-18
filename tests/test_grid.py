from __future__ import annotations

import numpy as np

from ridgeflow import GridSpec


def test_pixel_world_roundtrip(grid: GridSpec) -> None:
    points = np.array([[0.1, 0.1], [4.0, 4.0], [7.9, 7.9]])
    for point in points:
        assert np.allclose(grid.to_world(grid.to_pixel(point)), point)


def test_cell_centres_map_to_integers(grid: GridSpec) -> None:
    centre = (np.array([3, 5]) + 0.5) * grid.cell
    assert np.allclose(grid.to_pixel(centre), [3.0, 5.0])


def test_index_matches_floor(grid: GridSpec) -> None:
    assert grid.index([0.0, 0.0]) == (0, 0)
    assert grid.index([7.99, 7.99]) == (63, 63)
    assert not grid.holds(grid.index([8.5, 1.0]))
