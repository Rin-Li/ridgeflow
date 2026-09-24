from __future__ import annotations

from pathlib import Path

import torch

from ridgeflow.grid import GridSpec
from ridgeflow.model.flow import FlowMatching
from ridgeflow.model.targets import ENDPOINT_SIGMA
from ridgeflow.model.unet import build_unet


def pick_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    return torch.device(device)


class RidgeModel:
    """Samples a batch of ridge heatmaps from a rectified-flow checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        grid: GridSpec | None = None,
        device: str = "cuda",
        steps: int = 4,
        eta: float = 0.5,
    ) -> None:
        self.grid = grid or GridSpec()
        self.device = pick_device(device)
        self.steps = steps
        self.eta = eta
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = build_unet(state["model_config"]).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.flow = FlowMatching(**state.get("flow_config", {}))

    def _endpoints(self, world_xy: torch.Tensor) -> torch.Tensor:
        pixel = world_xy / self.grid.cell - 0.5
        axis = torch.arange(self.grid.size, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        dx = xx[None] - pixel[:, 0, None, None]
        dy = yy[None] - pixel[:, 1, None, None]
        return torch.exp(-(dx**2 + dy**2) / (2.0 * ENDPOINT_SIGMA**2))[:, None]

    @torch.no_grad()
    def sample(
        self, occupancy: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor, seed: int = 0
    ) -> torch.Tensor:
        """``(B, S, S)`` ``[x, y]`` occupancy -> ``(B, S, S)`` ``[x, y]`` heatmaps."""
        obstacle = occupancy.to(self.device, torch.float32).transpose(1, 2)[:, None]
        sampled = self.flow.sample(
            self.model,
            obstacle,
            self._endpoints(starts.to(self.device)),
            self._endpoints(goals.to(self.device)),
            steps=self.steps,
            eta=self.eta,
            seed=seed,
        )
        return sampled[:, 0].transpose(1, 2).contiguous()
