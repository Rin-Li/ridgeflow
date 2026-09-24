from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ridgeflow.collision import segments_blocked
from ridgeflow.grid import GridSpec
from ridgeflow.model.targets import RIDGE_SIGMA
from ridgeflow.planners.common import Result, collect

ACTIVE, ARRIVED, STUCK = 0, 1, 2


@dataclass(frozen=True)
class WalkerConfig:
    """Nothing is fitted: footprint = training ridge width, step = one cell, the cone only
    forbids stepping backwards, and ``truncation`` is a numerical tolerance."""

    step_px: float = 1.0
    cone_deg: float = 90.0
    turn_samples: int = 25
    sigma: float = RIDGE_SIGMA
    truncation: float = 1e-3
    max_steps: int = 1000

    @property
    def window_radius(self) -> int:
        return math.ceil(self.sigma * math.sqrt(2.0 * math.log(1.0 / self.truncation)) + 0.5)

    @property
    def amplitude(self) -> float:
        return self.step_px / (self.sigma * math.sqrt(2.0 * math.pi))


class RidgeWalker:
    """Greedy, feasibility-constrained matching pursuit over a ridge heatmap.

    x_{k+1} = argmax_{c in F_k} <R_k, Phi_c>,   R_{k+1} = max(R_k - a Phi_{x_{k+1}}, 0)
    """

    def __init__(self, grid: GridSpec | None = None, config: WalkerConfig | None = None) -> None:
        self.grid = grid or GridSpec()
        self.config = config or WalkerConfig()

    def _atoms(self, centres: torch.Tensor):
        radius = self.config.window_radius
        offsets = torch.arange(-radius, radius + 1, device=centres.device)
        base = torch.round(centres).long()
        ix = base[..., 0, None] + offsets
        iy = base[..., 1, None] + offsets
        scale = 2.0 * self.config.sigma**2
        gx = torch.exp(-((ix - centres[..., 0, None]) ** 2) / scale)
        gy = torch.exp(-((iy - centres[..., 1, None]) ** 2) / scale)
        top = self.grid.size + 2 * radius - 1
        return (ix + radius).clamp(0, top), (iy + radius).clamp(0, top), gx, gy

    def _consume(self, residual: torch.Tensor, centres: torch.Tensor, mask: torch.Tensor) -> None:
        rows = torch.nonzero(mask).squeeze(1)
        if len(rows) == 0:
            return
        ix, iy, gx, gy = self._atoms(centres[rows, None])
        ix, iy, gx, gy = ix[:, 0], iy[:, 0], gx[:, 0], gy[:, 0]
        index = (rows[:, None, None], ix[:, :, None], iy[:, None, :])
        atom = self.config.amplitude * gx[:, :, None] * gy[:, None, :]
        residual[index] = (residual[index] - atom).clamp_min(0.0)

    def _world(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels + 0.5) * self.grid.cell

    @torch.no_grad()
    def walk(
        self,
        heatmaps: torch.Tensor,
        grids: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
    ) -> Result:
        """``(B, S, S)`` heatmaps and planning grids, ``(B, 2)`` world endpoints."""
        config, cell = self.config, self.grid.cell
        device = heatmaps.device
        batch = heatmaps.shape[0]
        rows = torch.arange(batch, device=device)
        radius = config.window_radius

        position = starts.to(device) / cell - 0.5
        goal = goals.to(device) / cell - 0.5
        heading = F.normalize(goal - position, dim=-1)
        residual = F.pad(heatmaps.float(), (radius, radius, radius, radius))
        status = torch.full((batch,), ACTIVE, device=device)
        lengths = torch.ones(batch, dtype=torch.long, device=device)
        trail = [position.clone()]

        angles = torch.deg2rad(
            torch.linspace(-config.cone_deg, config.cone_deg, config.turn_samples, device=device)
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        self._consume(residual, position, status == ACTIVE)

        for _ in range(config.max_steps):
            active = status == ACTIVE
            if not bool(active.any()):
                break

            near = active & ((goal - position).norm(dim=-1) <= config.step_px)
            if bool(near.any()):
                free = ~segments_blocked(
                    grids, self._world(position), self._world(goal)[:, None], cell
                )[:, 0]
                arrive = near & free
                position = torch.where(arrive[:, None], goal, position)
                status = torch.where(arrive, ARRIVED, status)
                lengths += arrive
                active = active & ~arrive

            headings = torch.stack(
                [
                    heading[:, None, 0] * cos - heading[:, None, 1] * sin,
                    heading[:, None, 1] * cos + heading[:, None, 0] * sin,
                ],
                dim=-1,
            )
            candidates = position[:, None] + config.step_px * headings
            feasible = ~segments_blocked(
                grids, self._world(position), self._world(candidates), cell
            )

            ix, iy, gx, gy = self._atoms(candidates)
            window = residual[rows[:, None, None, None], ix[..., :, None], iy[..., None, :]]
            energy = torch.einsum("bnij,bni,bnj->bn", window, gx, gy)
            energy = energy.masked_fill(~feasible, float("-inf"))

            stuck = active & ~feasible.any(dim=1)
            status = torch.where(stuck, STUCK, status)
            active = active & ~stuck

            best = energy.argmax(dim=1)
            position = torch.where(active[:, None], candidates[rows, best], position)
            heading = torch.where(active[:, None], headings[rows, best], heading)
            lengths += active
            self._consume(residual, position, active)
            trail.append(position.clone())

        radius_slice = slice(radius, -radius)
        return Result(
            paths=collect([self._world(p) for p in trail], lengths),
            reached=(status == ARRIVED).cpu().numpy(),
            residual=residual[:, radius_slice, radius_slice],
        )
