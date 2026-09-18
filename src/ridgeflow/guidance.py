from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from ridgeflow.flow import FlowMatching
from ridgeflow.grid import GridSpec
from ridgeflow.targets import ENDPOINT_SIGMA, point_heatmap
from ridgeflow.unet import build_unet


class Guidance(Protocol):
    """Anything that turns a frozen occupancy snapshot into a ridge heatmap."""

    def heatmap(self, occupancy: np.ndarray, start_world, goal_world, seed: int) -> np.ndarray:
        ...

    @property
    def ms_per_call(self) -> float:
        ...


class FlowMatchingGuidance:
    """Samples the ridge field from a trained rectified-flow checkpoint.

    The model sees a static occupancy grid and knows nothing about velocities; the
    simulation's refresh rate is what keeps that snapshot current.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        grid: GridSpec | None = None,
        device: str = "cuda",
        steps: int = 4,
        eta: float = 0.5,
        endpoint_sigma: float = ENDPOINT_SIGMA,
    ) -> None:
        self.grid = grid or GridSpec()
        available = torch.cuda.is_available() or device == "cpu"
        self.device = torch.device(device if available else "cpu")
        self.steps = int(steps)
        self.eta = float(eta)
        self.endpoint_sigma = float(endpoint_sigma)

        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = build_unet(state["model_config"]).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.flow = FlowMatching(**state.get("flow_config", {}))

        self.calls = 0
        self.seconds = 0.0

    def _endpoint(self, world_xy) -> torch.Tensor:
        pixel = np.asarray(world_xy, np.float32) / self.grid.cell - 0.5
        return point_heatmap(pixel, self.grid.size, self.endpoint_sigma, self.device)[None]

    @torch.no_grad()
    def heatmap(self, occupancy: np.ndarray, start_world, goal_world, seed: int = 0) -> np.ndarray:
        """Occupancy arrives as ``[x, y]``; the network works in the ``[y, x]`` raster."""
        obstacle = torch.from_numpy(
            np.ascontiguousarray(np.asarray(occupancy, np.float32).T)
        )[None, None].to(self.device)
        start = self._endpoint(start_world)
        goal = self._endpoint(goal_world)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        sampled = self.flow.sample(
            self.model, obstacle, start, goal, steps=self.steps, eta=self.eta, seed=seed
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - began
        self.calls += 1

        raster = sampled[0, 0].float().cpu().numpy()
        return np.ascontiguousarray(raster.T)

    @property
    def ms_per_call(self) -> float:
        return 1000.0 * self.seconds / max(self.calls, 1)
