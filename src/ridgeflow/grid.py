from __future__ import annotations

from dataclasses import dataclass

import torch


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

    def cells(self, world_xy: torch.Tensor) -> torch.Tensor:
        """``(..., 2)`` world points -> the ``(..., 2)`` long indices of their cells."""
        return torch.floor(world_xy / self.cell).long()
