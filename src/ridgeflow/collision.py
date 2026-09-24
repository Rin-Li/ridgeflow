from __future__ import annotations

import torch

_JITTER = 0.1 * torch.tensor(
    [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
        [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5],
    ]
)


def blocked(grids: torch.Tensor, points: torch.Tensor, cell: float) -> torch.Tensor:
    """``(B, S, S)`` grids, ``(B, M, 2)`` world points -> ``(B, M)``; off-grid is blocked."""
    size = grids.shape[-1]
    rows = torch.arange(grids.shape[0], device=grids.device)[:, None]

    index = torch.floor(points / cell).long()
    outside = ((index < 0) | (index >= size)).any(-1)
    index = index.clamp(0, size - 1)
    hit = grids[rows, index[..., 0], index[..., 1]]

    jitter = _JITTER.to(points)
    probes = torch.floor((points[:, :, None] + jitter * cell) / cell).long()
    inside = ((probes >= 0) & (probes < size)).all(-1)
    probes = probes.clamp(0, size - 1)
    near = (grids[rows[:, :, None], probes[..., 0], probes[..., 1]] & inside).any(-1)
    return outside | hit | near


def segments_blocked(
    grids: torch.Tensor, origins: torch.Tensor, targets: torch.Tensor, cell: float
) -> torch.Tensor:
    """``(B, 2)`` origins, ``(B, N, 2)`` targets -> ``(B, N)``, sampled every quarter cell."""
    longest = float((targets - origins[:, None]).norm(dim=-1).max())
    count = max(int(longest / (0.25 * cell)) + 1, 10)
    ts = torch.linspace(0.0, 1.0, count, device=grids.device)[None, None, :, None]
    points = origins[:, None, None] + ts * (targets - origins[:, None])[:, :, None]
    batch, candidates = targets.shape[:2]
    hits = blocked(grids, points.reshape(batch, -1, 2), cell)
    return hits.view(batch, candidates, count).any(-1)
