from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ridgeflow.collision import blocked
from ridgeflow.grid import GridSpec
from ridgeflow.planners.common import Result, collect


@dataclass(frozen=True)
class DWAConfig:
    """A unicycle with the usual DWA cost: heading to goal, obstacle clearance, speed."""

    max_speed: float = 1.0
    min_speed: float = 0.0
    max_yaw_rate: float = math.radians(120.0)
    max_accel: float = 2.0
    max_yaw_accel: float = math.radians(360.0)
    dt: float = 0.1
    horizon: float = 0.6
    speed_samples: int = 9
    yaw_samples: int = 21
    heading_gain: float = 0.08
    clearance_gain: float = 0.3
    speed_gain: float = 3.0
    goal_tolerance: float = 0.2
    max_steps: int = 600


def clearance_field(grids: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    """Distance from each cell centre to the nearest blocked cell or the map edge."""
    size, cell = grid.size, grid.cell
    axis = (torch.arange(size, device=grids.device, dtype=torch.float32) + 0.5) * cell
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    centres = torch.stack([xx, yy], -1).view(-1, 2)
    edge = torch.minimum(
        torch.minimum(centres[:, 0], grid.extent - centres[:, 0]),
        torch.minimum(centres[:, 1], grid.extent - centres[:, 1]),
    )
    fields = []
    for occupied in grids:
        walls = centres[occupied.view(-1)]
        nearest = torch.cdist(centres, walls).amin(1) if len(walls) else edge
        fields.append(torch.minimum(nearest, edge).view(size, size))
    return torch.stack(fields)


class DWA:
    def __init__(self, grid: GridSpec | None = None, config: DWAConfig | None = None) -> None:
        self.grid = grid or GridSpec()
        self.config = config or DWAConfig()

    @torch.no_grad()
    def run(self, grids: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor) -> Result:
        c, cell = self.config, self.grid.cell
        device = grids.device
        batch = grids.shape[0]
        rows = torch.arange(batch, device=device)
        clearance = clearance_field(grids, self.grid)

        position = starts.to(device).clone()
        goal = goals.to(device)
        offset = goal - position
        yaw = torch.atan2(offset[:, 1], offset[:, 0])
        speed = torch.zeros(batch, device=device)
        turn = torch.zeros(batch, device=device)
        reached = torch.zeros(batch, dtype=torch.bool, device=device)
        lengths = torch.ones(batch, dtype=torch.long, device=device)
        trail = [position.clone()]

        steps = round(c.horizon / c.dt)
        k = torch.arange(1, steps + 1, device=device, dtype=torch.float32)
        unit_v = torch.linspace(0.0, 1.0, c.speed_samples, device=device)
        unit_w = torch.linspace(0.0, 1.0, c.yaw_samples, device=device)

        for _ in range(c.max_steps):
            active = ~reached
            if not bool(active.any()):
                break

            v_lo = (speed - c.max_accel * c.dt).clamp(c.min_speed, c.max_speed)
            v_hi = (speed + c.max_accel * c.dt).clamp(c.min_speed, c.max_speed)
            w_lo = (turn - c.max_yaw_accel * c.dt).clamp(-c.max_yaw_rate, c.max_yaw_rate)
            w_hi = (turn + c.max_yaw_accel * c.dt).clamp(-c.max_yaw_rate, c.max_yaw_rate)
            v = (v_lo[:, None] + (v_hi - v_lo)[:, None] * unit_v)[:, :, None]
            w = (w_lo[:, None] + (w_hi - w_lo)[:, None] * unit_w)[:, None, :]
            v, w = torch.broadcast_tensors(v, w)
            v, w = v.reshape(batch, -1), w.reshape(batch, -1)

            yaws = yaw[:, None, None] + w[..., None] * c.dt * k
            step = v[..., None] * c.dt
            xs = position[:, 0, None, None] + torch.cumsum(step * torch.cos(yaws), -1)
            ys = position[:, 1, None, None] + torch.cumsum(step * torch.sin(yaws), -1)
            points = torch.stack([xs, ys], -1)

            hits = blocked(grids, points.view(batch, -1, 2), cell).view(points.shape[:3]).any(-1)
            index = torch.floor(points / cell).long().clamp(0, self.grid.size - 1)
            gap = clearance[rows[:, None, None], index[..., 0], index[..., 1]].amin(-1)

            to_goal = goal[:, None] - points[:, :, -1]
            error = torch.atan2(to_goal[..., 1], to_goal[..., 0]) - yaws[..., -1]
            error = torch.atan2(torch.sin(error), torch.cos(error)).abs()
            cost = (
                c.heading_gain * error
                + c.clearance_gain / gap.clamp_min(1e-3)
                + c.speed_gain * (c.max_speed - v)
            ).masked_fill(hits, float("inf"))

            best = cost.argmin(dim=1)
            safe = torch.isfinite(cost[rows, best])
            chosen_v = torch.where(safe, v[rows, best], torch.zeros_like(speed))
            chosen_w = torch.where(safe, w[rows, best], torch.zeros_like(turn))
            speed = torch.where(active, chosen_v, speed)
            turn = torch.where(active, chosen_w, turn)
            yaw = torch.where(active, yaw + turn * c.dt, yaw)
            move = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1) * (speed * c.dt)[:, None]
            position = torch.where(active[:, None], position + move, position)
            lengths += active
            trail.append(position.clone())
            reached |= active & ((goal - position).norm(dim=-1) <= c.goal_tolerance)

        return Result(paths=collect(trail, lengths), reached=reached.cpu().numpy())
