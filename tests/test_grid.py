from __future__ import annotations

from ridgeflow import GridSpec


def test_index_matches_floor(grid: GridSpec) -> None:
    assert grid.cell == 0.125
    assert grid.index([0.0, 0.0]) == (0, 0)
    assert grid.index([7.99, 7.99]) == (63, 63)
