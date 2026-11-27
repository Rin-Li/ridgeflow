from __future__ import annotations

import numpy as np
import torch

from ridgeflow.grid import GridSpec

RIDGE_SIGMA = 1.5
ENDPOINT_SIGMA = 2.0


def world_to_pixel(points, grid: GridSpec) -> np.ndarray:
    pixels = np.asarray(points, np.float32) / grid.cell - 0.5
    return np.clip(pixels, 0.0, grid.size - 1)


def _mesh(size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def endpoint_heatmaps(pixels: torch.Tensor, size: int, sigma: float = ENDPOINT_SIGMA):
    """``(B, 2)`` pixel positions -> ``(B, 1, S, S)`` Gaussian blobs indexed ``[y, x]``."""
    xx, yy = _mesh(size, pixels.device)
    dx = xx - pixels[:, 0, None, None]
    dy = yy - pixels[:, 1, None, None]
    return torch.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))[:, None]


def path_heatmap(
    path_xy, size: int, sigma: float = RIDGE_SIGMA, chunk: int = 32, device="cpu"
) -> torch.Tensor:
    """A Gaussian tube of width ``sigma`` around the polyline, as ``(1, S, S)`` [y, x].

    ``sigma <= 0`` rasterises the polyline instead: every cell it passes through is 1."""
    points = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"expected a (L, 2) path, got {tuple(points.shape)}")
    if sigma <= 0:
        return _raster(points, size, device)
    if len(points) == 0:
        return torch.zeros((1, size, size), dtype=torch.float32, device=device)
    if len(points) == 1:
        return endpoint_heatmaps(points, size, sigma)[0]

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


def _raster(points: torch.Tensor, size: int, device) -> torch.Tensor:
    out = torch.zeros((1, size, size), dtype=torch.float32, device=device)
    if len(points) == 1:
        dense = points
    else:
        seg = points[1:] - points[:-1]
        steps = int(torch.ceil(seg.norm(dim=-1).max() / 0.25).item()) + 1
        t = torch.linspace(0.0, 1.0, steps, device=device)[None, :, None]
        dense = (points[:-1, None] + t * seg[:, None]).reshape(-1, 2)
    cells = torch.round(dense).long().clamp(0, size - 1)
    out[0, cells[:, 1], cells[:, 0]] = 1.0
    return out


def path_distance(path_xy, size: int, chunk: int = 32, device="cpu") -> torch.Tensor:
    """Euclidean distance from every pixel centre to the polyline, ``(S, S)`` [y, x]."""
    points = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    xx, yy = _mesh(size, device)
    if len(points) == 1:
        return ((xx - points[0, 0]) ** 2 + (yy - points[0, 1]) ** 2).sqrt()
    nearest = torch.full((size, size), float("inf"), dtype=torch.float32, device=device)
    heads, deltas = points[:-1], points[1:] - points[:-1]
    for begin in range(0, len(heads), chunk):
        ax = heads[begin : begin + chunk, 0, None, None]
        ay = heads[begin : begin + chunk, 1, None, None]
        dx = deltas[begin : begin + chunk, 0, None, None]
        dy = deltas[begin : begin + chunk, 1, None, None]
        length = (dx.square() + dy.square()).clamp_min(1e-12)
        t = (((xx[None] - ax) * dx + (yy[None] - ay) * dy) / length).clamp(0.0, 1.0)
        d2 = (xx[None] - (ax + t * dx)).square() + (yy[None] - (ay + t * dy)).square()
        nearest = torch.minimum(nearest, d2.amin(dim=0))
    return nearest.sqrt()


@torch.no_grad()
def geodesic_ridges(paths_px, blocked: torch.Tensor, sigma: float = RIDGE_SIGMA) -> torch.Tensor:
    """Ridges whose distance is measured inside free space: ``(B, S, S)`` [y, x].

    Cells within one pixel of the path start at their Euclidean distance to it; the distance then
    grows only through free cells (8-connected, no corner cutting, octile steps), so a ridge
    never leaks through a wall however thin, and obstacle cells are 0. In open space it matches
    ``path_heatmap``. ``blocked`` is ``(B, S, S)`` [y, x] on the device to compute on.
    """
    from ridgeflow.routes import distances_from_field, edge_weights

    device = blocked.device
    free = ~blocked.bool()
    near = torch.stack([path_distance(p, blocked.shape[-1], device=device) for p in paths_px])
    seed = torch.where((near <= 1.0) & free, near, torch.full_like(near, float("inf")))
    weights = edge_weights(torch.ones_like(near), free)
    dist = distances_from_field(weights, seed)
    if sigma <= 0:
        return (dist <= 0.5).float()
    return torch.where(free, torch.exp(-dist.square() / (2.0 * sigma**2)), torch.zeros_like(dist))
