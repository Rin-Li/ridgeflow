from __future__ import annotations

import numpy as np
import torch

from ridgeflow.grid import GridSpec

RIDGE_SIGMA = 1.5
ENDPOINT_SIGMA = 2.0


def world_to_pixel(points, grid: GridSpec) -> np.ndarray:
    """World coordinates to fractional pixel centres, clipped to the grid."""
    pixels = np.asarray(points, np.float32) / grid.cell - 0.5
    return np.clip(pixels, 0.0, grid.size - 1)


def _mesh(size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def point_heatmap(
    pixel_xy, size: int, sigma: float = ENDPOINT_SIGMA, device="cpu"
) -> torch.Tensor:
    """A single Gaussian blob as a ``(1, size, size)`` raster indexed ``[y, x]``."""
    point = torch.as_tensor(pixel_xy, dtype=torch.float32, device=device)
    xx, yy = _mesh(size, device)
    field = torch.exp(-((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / (2.0 * sigma**2))
    return field.unsqueeze(0)


def path_heatmap(
    path_xy, size: int, sigma: float = RIDGE_SIGMA, chunk: int = 32, device="cpu"
) -> torch.Tensor:
    """A Gaussian tube around the piecewise-linear path, measured to segments.

    Distance is taken to every segment rather than to the stored waypoints, so the
    target does not depend on how densely the planner sampled the path.
    """
    points = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"expected a (L, 2) path, got {tuple(points.shape)}")
    if len(points) == 0:
        return torch.zeros((1, size, size), dtype=torch.float32, device=device)
    if len(points) == 1:
        return point_heatmap(points[0], size, sigma, device)

    xx, yy = _mesh(size, device)
    nearest = torch.full((size, size), float("inf"), dtype=torch.float32, device=device)
    heads = points[:-1]
    deltas = points[1:] - heads
    for begin in range(0, len(heads), chunk):
        anchor = heads[begin : begin + chunk]
        delta = deltas[begin : begin + chunk]
        ax, ay = anchor[:, 0, None, None], anchor[:, 1, None, None]
        dx, dy = delta[:, 0, None, None], delta[:, 1, None, None]
        length = (dx.square() + dy.square()).clamp_min(1e-12)
        t = (((xx[None] - ax) * dx + (yy[None] - ay) * dy) / length).clamp(0.0, 1.0)
        offset_x = xx[None] - (ax + t * dx)
        offset_y = yy[None] - (ay + t * dy)
        nearest = torch.minimum(nearest, (offset_x.square() + offset_y.square()).amin(dim=0))
    return torch.exp(-nearest / (2.0 * sigma**2)).unsqueeze(0)
