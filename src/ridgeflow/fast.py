"""Faster drop-ins for ``RidgeModel`` and ``RidgeWalker`` (kept apart from the defaults for now).

``FastRidgeModel``  the same sampler as ``FlowMatching.sample`` (same noise for the same seed),
                    with the UNet under bf16 autocast and the whole few-step loop compiled by
                    ``torch.compile`` (fused kernels, CUDA graphs). bf16 changes the numbers
                    slightly, so a draw can end on a different route than the fp32 draw of the
                    same seed; ``dtype=None, compile=False`` reproduces ``RidgeModel`` exactly.
                    The first call per batch size compiles (about a minute): call ``warmup``.
``FastRidgeWalker`` the same walks as ``RidgeWalker``, bit for bit, from the kernels in
                    ``planners._walker_numba_fast``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ridgeflow.grid import GridSpec
from ridgeflow.model.guidance import RidgeModel
from ridgeflow.planners.common import Result
from ridgeflow.planners.walker import RidgeWalker


class FastRidgeModel(RidgeModel):
    def __init__(
        self,
        checkpoint: str | Path,
        grid: GridSpec | None = None,
        device: str = "cuda",
        steps: int = 4,
        eta: float = 0.5,
        dtype: torch.dtype | None = torch.bfloat16,
        compile: bool = True,
        compile_mode: str = "max-autotune",
    ) -> None:
        super().__init__(checkpoint, grid, device, steps, eta)
        self.dtype = dtype if self.device.type == "cuda" else None
        self._schedule = [1.0 - i / steps for i in range(steps + 1)]
        self._run = torch.compile(self._loop, mode=compile_mode) if compile else self._loop

    def _loop(self, x, obstacle, start, goal, noises):
        """``FlowMatching.sample`` (rho 1) with its noise drawn up front."""
        batch = x.shape[0]
        for i in range(self.steps):
            t, t_next = self._schedule[i], self._schedule[i + 1]
            tt = torch.full((batch,), t * self.flow.time_scale, device=x.device)
            with torch.autocast(
                "cuda", dtype=self.dtype or torch.float32, enabled=bool(self.dtype)
            ):
                velocity = self.model(x, obstacle, start, goal, tt)
            velocity = velocity.float()
            deterministic = x - velocity * (t - t_next)
            if self.eta > 0.0 and t_next > 0.0:
                x0 = x - t * velocity
                stochastic = (1.0 - t_next) * x0 + t_next * noises[i]
                x = (1.0 - self.eta) * deterministic + self.eta * stochastic
            else:
                x = deterministic
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(
        self, occupancy: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor, seed: int = 0
    ) -> torch.Tensor:
        """``(B, S, S)`` ``[x, y]`` occupancy -> ``(B, S, S)`` ``[x, y]`` heatmaps."""
        obstacle = occupancy.to(self.device, torch.float32).transpose(1, 2)[:, None]
        shape = (obstacle.shape[0], 1, obstacle.shape[-2], obstacle.shape[-1])
        torch.manual_seed(seed)  # same draws, in the same order, as FlowMatching.sample
        x = torch.randn(shape, device=self.device)
        noises = torch.stack(
            [
                torch.randn(shape, device=self.device)
                if self.eta > 0.0 and self._schedule[i + 1] > 0.0
                else torch.zeros(shape, device=self.device)
                for i in range(self.steps)
            ]
        )
        out = self._run(x, obstacle, self._endpoints(starts), self._endpoints(goals), noises)
        # clone: with CUDA graphs the output buffer is reused by the next call
        return out[:, 0].transpose(1, 2).contiguous().clone()

    def warmup(self, batch_sizes=(1, 4, 8, 16)) -> None:
        """Compile (and capture) every batch size used later, so no query pays for it."""
        s = self.grid.size
        for b in batch_sizes:
            occ = torch.zeros(b, s, s, dtype=torch.bool, device=self.device)
            ends = torch.full((b, 2), self.grid.extent / 2, device=self.device)
            for _ in range(3):
                self.sample(occ, ends, ends)
        if self.device.type == "cuda":
            torch.cuda.synchronize()


class FastRidgeWalker(RidgeWalker):
    def _walk_numba(self, energy_map, grids, starts, goals) -> Result:
        from ridgeflow.planners._walker_numba_fast import (
            _JITTER,
            ARRIVED,
            walk_batch,
            walk_batch_dfs,
        )

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
            trail, lengths, status, residual, backtracks = walk_batch_dfs(*args, config.lost_floor)
        else:
            trail, lengths, status, residual = walk_batch(*args)
        paths = [((trail[b, :n] + 0.5) * cell).astype(np.float32) for b, n in enumerate(lengths)]
        crop = residual[:, radius:-radius, radius:-radius]
        return Result(
            paths=paths,
            reached=status == ARRIVED,
            residual=torch.from_numpy(crop).float().to(energy_map.device),
            backtracks=backtracks,
        )
