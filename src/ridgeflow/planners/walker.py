from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ridgeflow.collision import segments_blocked
from ridgeflow.grid import GridSpec
from ridgeflow.model.targets import RIDGE_SIGMA
from ridgeflow.planners.common import Result, collect

ACTIVE, ARRIVED, STUCK = 0, 1, 2


def _use_numba(backend: str) -> bool:
    if backend == "torch":
        return False
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise
        return False
    return True


@dataclass(frozen=True)
class WalkerConfig:
    """Nothing is fitted: see the constants table in the README."""

    step_px: float = 1.0
    cone_deg: float = 90.0
    turn_samples: int = 25
    sigma: float = RIDGE_SIGMA
    truncation: float = 1e-3
    max_steps: int = 1000
    arrive_px: float | None = None  # None: one ridge width (2 sigma)
    reverse_when_stuck: bool = True  # every forward candidate blocked: try the rear sector
    repair: bool = False  # finish failed walks along the ridge geodesic, string-pulled
    backtrack: bool = True  # remember branch points; return to them when stuck or off the ridge
    lost_energy: float | None = None  # below this the walker has left the ridge (None: tau x atom)
    backend: str = "auto"  # "numba" (CPU, compiled), "torch", or "auto": numba when installed

    @property
    def window_radius(self) -> int:
        return math.ceil(self.sigma * math.sqrt(2.0 * math.log(1.0 / self.truncation)) + 0.5)

    @property
    def forward_deg(self) -> np.ndarray:
        return np.linspace(-self.cone_deg, self.cone_deg, self.turn_samples)

    @property
    def rear_deg(self) -> np.ndarray:
        """The rest of the circle at the forward cone's spacing; empty if reversing is off."""
        if not self.reverse_when_stuck:
            return np.zeros(0)
        gap = 2.0 * self.cone_deg / (self.turn_samples - 1)
        return np.arange(self.cone_deg + gap, 360.0 - self.cone_deg - gap / 2, gap)

    @property
    def atom_mass(self) -> float:
        r = np.arange(-self.window_radius, self.window_radius + 1)
        return float(np.exp(-(r**2) / (2.0 * self.sigma**2)).sum() ** 2)

    @property
    def lost_floor(self) -> float:
        """Energy of a ridge at the truncation level, unless set explicitly."""
        if self.lost_energy is not None:
            return self.lost_energy
        return self.truncation * self.atom_mass

    @property
    def arrival_radius(self) -> float:
        """Connect to the goal once it is within one ridge width and the segment is free."""
        return 2.0 * self.sigma if self.arrive_px is None else self.arrive_px

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

    def _walk_numba(self, energy_map, grids, starts, goals) -> Result:
        from ridgeflow.planners._walker_numba import _JITTER, walk_batch, walk_batch_dfs
        from ridgeflow.planners._walker_numba import ARRIVED as DONE

        config, cell = self.config, self.grid.cell
        radius = config.window_radius
        args = (
            energy_map.detach().cpu().numpy().astype(np.float64),
            grids.detach().cpu().numpy().astype(np.bool_),
            starts.detach().cpu().numpy().astype(np.float64) / cell - 0.5,
            goals.detach().cpu().numpy().astype(np.float64) / cell - 0.5,
            cell, config.step_px, np.deg2rad(config.forward_deg), np.deg2rad(config.rear_deg),
            config.sigma, radius, config.amplitude, config.arrival_radius, config.max_steps,
            _JITTER,
        )  # fmt: skip
        backtracks = None
        if config.backtrack:
            trail, lengths, status, residual, backtracks = walk_batch_dfs(
                *args, config.lost_floor
            )
        else:
            trail, lengths, status, residual = walk_batch(*args)
        paths = [((trail[b, :n] + 0.5) * cell).astype(np.float32) for b, n in enumerate(lengths)]
        crop = residual[:, radius:-radius, radius:-radius]
        return Result(
            paths=paths,
            reached=status == DONE,
            residual=torch.from_numpy(crop).float().to(energy_map.device),
            backtracks=backtracks,
        )

    def _world(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels + 0.5) * self.grid.cell

    @torch.no_grad()
    def walk(
        self,
        heatmaps: torch.Tensor,
        grids: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        route: torch.Tensor | None = None,
    ) -> Result:
        """``(B, S, S)`` heatmaps and planning grids, ``(B, 2)`` world endpoints.

        ``route``, optional ``(B, S, S)`` cells of one feasible start-goal route (the L2
        projection's): the walker then only reads ridge energy within its own window radius of
        that route, so bright ridge the model drew into a dead end cannot lure it away.

        With ``config.repair`` a walk that fails is finished from the point of its trail that is
        cheapest to the goal, along the ridge geodesic (cost ``1 - log max(H, tau)``), with every
        collision-free shortcut taken. That route is feasible by construction, so every query
        whose goal is reachable on the grid (8-connected, no corner cutting) succeeds.
        """
        result = self._walk_greedy(heatmaps, grids, starts, goals, route)
        if self.config.repair:
            result = self._repair(result, heatmaps, grids, starts, goals)
        return result

    def _repair(self, result: Result, heatmaps, grids, starts, goals) -> Result:
        from ridgeflow.routes import distances, edge_weights, ridge_cost

        reached = np.asarray(result.reached, bool).copy()
        repaired = np.zeros_like(reached)
        failed = np.flatnonzero(~reached)
        if len(failed) == 0:
            result.repaired = repaired
            return result
        cell, size = self.grid.cell, self.grid.size
        dev = heatmaps.device
        idx = torch.as_tensor(failed, device=dev)
        free = ~grids.to(dev)[idx]
        goal_cells = torch.floor(goals.to(dev)[idx] / cell).long().clamp(0, size - 1)
        rows = torch.arange(len(failed), device=dev)
        free[rows, goal_cells[:, 0], goal_cells[:, 1]] = True
        cost_map = ridge_cost(heatmaps[idx].clamp(0, 1))
        weights = edge_weights(cost_map, free)
        to_goal = distances(weights, goal_cells).cpu().numpy()
        cost_np = cost_map.cpu().numpy()
        w = weights.cpu().numpy()
        grids_np = grids.detach().cpu().numpy().astype(bool)
        paths = list(result.paths)
        for k, b in enumerate(failed):
            trail = np.asarray(paths[b], np.float64)
            cells = np.clip(np.floor(trail / cell).astype(int), 0, size - 1)
            remaining = to_goal[k, cells[:, 0], cells[:, 1]]
            if not np.isfinite(remaining).any():
                continue  # the goal is not reachable on the grid from anywhere it went
            # splice where the whole path is cheapest under the same ridge metric: the trail so
            # far (step length x ridge cost) plus the geodesic from there, so wandering off the
            # ridge before the splice point is paid for and cut away
            step = np.linalg.norm(np.diff(trail, axis=0), axis=1) / cell
            here = cost_np[k, cells[:, 0], cells[:, 1]]
            walked = np.concatenate([[0.0], np.cumsum(step * 0.5 * (here[1:] + here[:-1]))])
            j = int(np.argmin(walked + remaining))
            cur = (int(cells[j, 0]), int(cells[j, 1]))
            goal_cell = (int(goal_cells[k, 0]), int(goal_cells[k, 1]))
            route = [cur]
            while cur != goal_cell:  # descend the distance field, one cell at a time
                best, nxt = math.inf, None
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = cur[0] + dx, cur[1] + dy
                        if (dx or dy) and 0 <= nx < size and 0 <= ny < size:
                            v = to_goal[k, nx, ny] + w[k, nx, ny, 1 - dx, 1 - dy]
                            if v < best:
                                best, nxt = v, (nx, ny)
                cur = nxt
                route.append(cur)
            tail = (np.asarray(route, np.float64) + 0.5) * cell
            tail[-1] = goals[b].detach().cpu().numpy()
            points = np.concatenate([trail[j : j + 1], tail])
            pulled = _string_pull(points, grids_np[b], cell)
            paths[b] = np.concatenate([trail[:j], pulled]).astype(np.float32)
            reached[b] = repaired[b] = True
        return Result(
            paths=paths, reached=reached, residual=result.residual, repaired=repaired,
            backtracks=result.backtracks,
        )

    @torch.no_grad()
    def _walk_greedy(
        self,
        heatmaps: torch.Tensor,
        grids: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        route: torch.Tensor | None = None,
    ) -> Result:
        config, cell = self.config, self.grid.cell
        device = heatmaps.device
        batch = heatmaps.shape[0]
        rows = torch.arange(batch, device=device)
        radius = config.window_radius

        position = starts.to(device) / cell - 0.5
        goal = goals.to(device) / cell - 0.5
        heading = F.normalize(goal - position, dim=-1)
        energy_map = heatmaps.float()
        if route is not None:
            k = 2 * radius + 1
            corridor = F.max_pool2d(route.float()[:, None], k, 1, radius)[:, 0]
            energy_map = energy_map * corridor
        if _use_numba(config.backend):
            return self._walk_numba(energy_map, grids, starts, goals)
        if config.backtrack:
            warnings.warn(
                "backtracking needs the numba backend; walking greedily", stacklevel=3
            )
        residual = F.pad(energy_map, (radius, radius, radius, radius))
        status = torch.full((batch,), ACTIVE, device=device)
        lengths = torch.ones(batch, dtype=torch.long, device=device)
        trail = [position.clone()]

        n_forward = config.turn_samples
        degrees = np.concatenate([config.forward_deg, config.rear_deg])
        angles = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32, device=device))
        cos, sin = torch.cos(angles), torch.sin(angles)
        self._consume(residual, position, status == ACTIVE)

        for _ in range(config.max_steps):
            active = status == ACTIVE
            if not bool(active.any()):
                break

            near = active & ((goal - position).norm(dim=-1) <= config.arrival_radius)
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
            # the rear sector only counts when every forward candidate is blocked
            forward_ok = feasible[:, :n_forward].any(dim=1, keepdim=True)
            energy[:, n_forward:] = energy[:, n_forward:].masked_fill(forward_ok, float("-inf"))
            feasible = feasible.clone()
            feasible[:, n_forward:] &= ~forward_ok

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


def _string_pull(points: np.ndarray, grid: np.ndarray, cell: float) -> np.ndarray:
    """Greedy any-angle shortcutting: from each kept point jump to the farthest later point in
    sight (collision-checked like every walker step). Consecutive points are kept as they are."""
    try:
        from ridgeflow.planners._walker_numba import _JITTER, _segment_blocked

        def blocked(a, b):
            return _segment_blocked(grid, a[0], a[1], b[0], b[1], cell, _JITTER)
    except ImportError:
        from ridgeflow.collision import segments_blocked

        g = torch.from_numpy(grid)[None]

        def blocked(a, b):
            a_t = torch.as_tensor(a, dtype=torch.float32)[None]
            b_t = torch.as_tensor(b, dtype=torch.float32)[None, None]
            return bool(segments_blocked(g, a_t, b_t, cell)[0, 0])

    kept, i, n = [points[0]], 0, len(points)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and blocked(points[i], points[j]):
            j -= 1
        kept.append(points[j])
        i = j
    return np.asarray(kept)
