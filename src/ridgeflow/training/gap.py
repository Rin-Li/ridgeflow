"""Gap loss: the ridge a model draws must not break along its own route.

Flow matching scores every cell alike, so a heatmap that is right almost everywhere but breaks
the ridge in one or two cells costs next to nothing, yet leaves the walker with no way through.
The gap loss puts the weight on exactly those cells.

For the one-step estimate ``x0_hat = x_t - t * v`` of the clean heatmap ``h``:

1. the ridge geodesic is the cheapest free start-goal route under ``c = 1 - log max(h, TAU)``
   per step (8-connected, no corner cutting). ``-log`` of a Gaussian ridge is the squared
   distance to its centreline, so on the ridge a step costs about one and leaving it costs more;
2. the gaps are the cells of that route where ``h < 1/2``;
3. ``L_gap = (1 - t) * mean_cells[(1 - h)^2 * gap]``.

The route is found on a detached copy of ``h``: the loss only lifts the gap cells, it does not
move the route. It is the model's own route, not the demonstration's, so the loss closes the
route the sample chose and does not pull samples towards one homotopy class. The ``1 - t``
weight fades the loss out at high noise, where ``x0_hat`` is a blur and its route means little.
"""

from __future__ import annotations

import torch

from ridgeflow.routes import ON_RIDGE, ridge_cost, route_mask

__all__ = ["ON_RIDGE", "gap_loss", "geodesic_mask"]


def geodesic_mask(h: torch.Tensor, free: torch.Tensor, start, goal) -> torch.Tensor:
    """``(B, S, S)`` cells on the ridge geodesic of ``h`` (all of them, where routes tie)."""
    return route_mask(ridge_cost(h), free, start, goal)


def gap_loss(
    x0_hat: torch.Tensor,
    obstacle: torch.Tensor,
    start_pixel: torch.Tensor,
    goal_pixel: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x0_hat`` and ``obstacle`` are ``(B, 1, S, S)`` [y, x]; pixels are ``(B, 2)`` [x, y];
    ``weight`` is ``(B,)``, the per-sample weight (``1 - t`` in training).

    Returns the loss (a per-cell mean, on the scale of the flow MSE) and the mean number of
    gap cells per map, for logging.
    """
    h = x0_hat[:, 0]
    start = torch.floor(start_pixel + 0.5).long().flip(-1)  # [x, y] -> [row, col]
    goal = torch.floor(goal_pixel + 0.5).long().flip(-1)
    with torch.no_grad():
        route = geodesic_mask(h.clamp(0.0, 1.0), obstacle[:, 0] < 0.5, start, goal)
        gaps = route & (h < ON_RIDGE)
    per_map = ((1.0 - h).square() * gaps).mean(dim=(1, 2))
    return (weight * per_map).mean(), gaps.float().sum(dim=(1, 2)).mean()
