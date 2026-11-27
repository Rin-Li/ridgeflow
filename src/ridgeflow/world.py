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
    """``occupancy`` is a conservative raster: a cell is blocked when any part of it overlaps
    an obstacle, so every free cell lies wholly in free space and a path that stays in free
    cells cannot touch an obstacle, with no padding needed. The model sees this raster and
    planners avoid it (padded by ``padding_px``, 0 by default)."""

    rects: list[Rect]
    start: np.ndarray
    goal: np.ndarray
    grid: GridSpec = field(default_factory=GridSpec)
    padding_px: int = 0
    _occupancy: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _planning: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def occupancy(self) -> torch.Tensor:
        if self._occupancy is None:
            low = torch.arange(self.grid.size, dtype=torch.float64) * self.grid.cell
            high = low + self.grid.cell
            raster = torch.zeros(self.grid.size, self.grid.size, dtype=torch.bool)
            for r in self.rects:  # positive-area overlap; touching an edge does not block
                xs = (low < r.x + r.width) & (high > r.x)
                ys = (low < r.y + r.height) & (high > r.y)
                raster |= xs[:, None] & ys[None, :]
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
        """Free on the planning raster and, exactly, clear of every obstacle rectangle."""
        path = np.asarray(path, np.float64)
        if len(path) < 2:
            return len(path) == 1 and not self.blocked(path[0])
        pairs = zip(path[:-1], path[1:], strict=True)
        if any(self.segment_blocked(a, b) for a, b in pairs):
            return False
        return not any(_segments_hit_rect(path[:-1], path[1:], r) for r in self.rects)


def _segments_hit_rect(a: np.ndarray, b: np.ndarray, r: Rect) -> bool:
    """Does any segment a[i]-b[i] meet the closed rectangle ``r``? (Liang-Barsky clipping)"""
    d = b - a
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    ok = np.ones(len(a), bool)
    for p, q in (
        (-d[:, 0], a[:, 0] - r.x),
        (d[:, 0], r.x + r.width - a[:, 0]),
        (-d[:, 1], a[:, 1] - r.y),
        (d[:, 1], r.y + r.height - a[:, 1]),
    ):
        parallel = p == 0
        ok &= ~(parallel & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(parallel, 0.0, q / np.where(parallel, 1.0, p))
        t0 = np.where(~parallel & (p < 0), np.maximum(t0, t), t0)
        t1 = np.where(~parallel & (p > 0), np.minimum(t1, t), t1)
    return bool((ok & (t0 <= t1)).any())


def stack(worlds: list[World], device) -> tuple[torch.Tensor, ...]:
    occupancy = torch.stack([w.occupancy() for w in worlds]).to(device)
    grids = torch.stack([w.planning_grid() for w in worlds]).to(device)
    starts = torch.tensor(np.stack([w.start for w in worlds]), dtype=torch.float32, device=device)
    goals = torch.tensor(np.stack([w.goal for w in worlds]), dtype=torch.float32, device=device)
    return occupancy, grids, starts, goals
