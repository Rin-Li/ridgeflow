from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ridgeflow.collision import blocked, segments_blocked
from ridgeflow.grid import GridSpec


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def overlaps(self, other: Rect) -> bool:
        return not (
            self.x + self.width < other.x
            or self.x > other.x + other.width
            or self.y + self.height < other.y
            or self.y > other.y + other.height
        )


@dataclass
class World:
    """``occupancy`` is what the model sees; planners avoid it padded by ``padding_px``."""

    rects: list[Rect]
    start: np.ndarray
    goal: np.ndarray
    grid: GridSpec = field(default_factory=GridSpec)
    padding_px: int = 1
    _occupancy: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _planning: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def occupancy(self) -> torch.Tensor:
        if self._occupancy is None:
            centres = (torch.arange(self.grid.size) + 0.5) * self.grid.cell
            xx, yy = torch.meshgrid(centres, centres, indexing="ij")
            raster = torch.zeros(self.grid.size, self.grid.size, dtype=torch.bool)
            for r in self.rects:
                raster |= (xx >= r.x) & (xx <= r.x + r.width) & (yy >= r.y) & (yy <= r.y + r.height)
            self._occupancy = raster
        return self._occupancy

    def planning_grid(self) -> torch.Tensor:
        if self._planning is None:
            raster = self.occupancy().float()[None, None]
            k = 2 * self.padding_px + 1
            self._planning = F.max_pool2d(raster, k, 1, self.padding_px)[0, 0] > 0.5
        return self._planning

    def blocked(self, point) -> bool:
        point = torch.as_tensor(np.asarray(point), dtype=torch.float32).view(1, 1, 2)
        return bool(blocked(self.planning_grid()[None], point, self.grid.cell)[0, 0])

    def segment_blocked(self, a, b) -> bool:
        a = torch.as_tensor(np.asarray(a), dtype=torch.float32).view(1, 2)
        b = torch.as_tensor(np.asarray(b), dtype=torch.float32).view(1, 1, 2)
        return bool(segments_blocked(self.planning_grid()[None], a, b, self.grid.cell)[0, 0])

    def path_free(self, path) -> bool:
        path = np.asarray(path, np.float64)
        if len(path) < 2:
            return len(path) == 1 and not self.blocked(path[0])
        pairs = zip(path[:-1], path[1:], strict=True)
        return not any(self.segment_blocked(a, b) for a, b in pairs)


def stack(worlds: list[World], device) -> tuple[torch.Tensor, ...]:
    occupancy = torch.stack([w.occupancy() for w in worlds]).to(device)
    grids = torch.stack([w.planning_grid() for w in worlds]).to(device)
    starts = torch.tensor(np.stack([w.start for w in worlds]), dtype=torch.float32, device=device)
    goals = torch.tensor(np.stack([w.goal for w in worlds]), dtype=torch.float32, device=device)
    return occupancy, grids, starts, goals
