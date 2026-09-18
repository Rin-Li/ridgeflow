from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Square occupancy grid covering ``[0, extent]^2`` world units."""

    size: int = 64
    extent: float = 8.0

    @property
    def cell(self) -> float:
        return self.extent / self.size

    def to_pixel(self, world_xy) -> np.ndarray:
        return np.asarray(world_xy, np.float64) / self.cell - 0.5

    def to_world(self, pixel_xy) -> np.ndarray:
        return (np.asarray(pixel_xy, np.float64) + 0.5) * self.cell

    def index(self, world_xy) -> tuple[int, int]:
        return int(np.floor(world_xy[0] / self.cell)), int(np.floor(world_xy[1] / self.cell))

    def holds(self, index: tuple[int, int]) -> bool:
        return 0 <= index[0] < self.size and 0 <= index[1] < self.size

    def clip(self, world_xy, margin: float = 1e-3) -> np.ndarray:
        return np.clip(np.asarray(world_xy, np.float64), margin, self.extent - margin)
