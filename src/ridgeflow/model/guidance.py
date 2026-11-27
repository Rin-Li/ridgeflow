from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ridgeflow.grid import GridSpec
from ridgeflow.model.flow import FlowMatching
from ridgeflow.model.targets import endpoint_heatmaps
from ridgeflow.model.unet import build_unet


def pick_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    return torch.device(device)


class _BFloat16(nn.Module):
    """Runs the wrapped network under bf16 autocast and hands back fp32."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(*args).float()


class RidgeModel:
    """Samples a batch of ridge heatmaps from a rectified-flow checkpoint.

    ``fast=True`` (CUDA only) runs the UNet under bf16 autocast and compiles the whole few-step
    loop with ``torch.compile``: about 3x faster for one query, 2x for a batch of 8. The noise is
    the same for the same seed, but bf16 rounding can send an occasional draw down a different
    route. The first call per batch size compiles for about a minute; ``warmup`` does it up
    front.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        grid: GridSpec | None = None,
        device: str = "cuda",
        steps: int = 4,
        eta: float = 0.5,
        fast: bool = False,
    ) -> None:
        self.grid = grid or GridSpec()
        self.device = pick_device(device)
        self.steps = steps
        self.eta = eta
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = build_unet(state["model_config"]).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.flow = FlowMatching(**state.get("flow_config", {}))
        self.fast = fast and self.device.type == "cuda"
        self._net = _BFloat16(self.model) if self.fast else self.model
        integrate = self.flow.integrate
        self._integrate = torch.compile(integrate, mode="max-autotune") if self.fast else integrate

    def _endpoints(self, world_xy: torch.Tensor) -> torch.Tensor:
        pixels = world_xy.to(self.device) / self.grid.cell - 0.5
        return endpoint_heatmaps(pixels, self.grid.size)

    @torch.no_grad()
    def sample(
        self, occupancy: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor, seed: int = 0
    ) -> torch.Tensor:
        """``(B, S, S)`` ``[x, y]`` occupancy -> ``(B, S, S)`` ``[x, y]`` heatmaps."""
        obstacle = occupancy.to(self.device, torch.float32).transpose(1, 2)[:, None]
        start, goal = self._endpoints(starts), self._endpoints(goals)
        shape = (obstacle.shape[0], 1, *obstacle.shape[-2:])
        x, kicks = self.flow.noise(shape, self.steps, self.eta, self.device, seed)
        x = self._integrate(self._net, x, kicks, obstacle, start, goal, self.steps, self.eta)
        # contiguous copies out of the buffer a compiled (CUDA graph) call reuses
        return x.clamp(0.0, 1.0)[:, 0].transpose(1, 2).contiguous()

    def warmup(self, batch_sizes=(1, 4, 8, 16)) -> None:
        """With ``fast``, compile every batch size used later, so no query pays for it."""
        if not self.fast:
            return
        s = self.grid.size
        for b in batch_sizes:
            occupancy = torch.zeros(b, s, s, dtype=torch.bool, device=self.device)
            ends = torch.full((b, 2), self.grid.extent / 2, device=self.device)
            for _ in range(3):
                self.sample(occupancy, ends, ends)
        torch.cuda.synchronize()
