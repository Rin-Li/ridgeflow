from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    """A square grid of ``size`` cells over ``[0, extent]^2``."""

    size: int = 64
    extent: float = 8.0

    @property
    def cell(self) -> float:
        return self.extent / self.size

    def index(self, world_xy) -> tuple[int, int]:
        return int(world_xy[0] // self.cell), int(world_xy[1] // self.cell)
