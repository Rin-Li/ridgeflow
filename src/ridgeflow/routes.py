"""Batched cheapest routes on the grid, for any per-cell cost.

A step between 8-neighbours costs its length times the mean of the two cells' costs; diagonal
steps may not cut a blocked corner. Everything is batched on the device; ``(B, S, S)`` tensors
are indexed the same way as ``start``/``goal`` ``(B, 2)`` long indices (either [x, y] or [y, x]).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

TAU = 1e-3  # the walker's truncation: below this a ridge is "nothing"
ON_RIDGE = 0.5
NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _view(padded: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """From a 1-padded ``(B, S+2, S+2)`` tensor, the neighbour at ``(-dx, -dy)`` of each cell."""
    s = padded.shape[-1] - 2
    return padded[:, 1 - dx : 1 - dx + s, 1 - dy : 1 - dy + s]


def edge_weights(cost: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """``(B, S, S, 3, 3)``: entry ``[i, j, 1 - dx, 1 - dy]`` is the cost of the step from
    cell ``(i - dx, j - dy)`` into ``(i, j)``; the centre is 0 so a cell keeps its distance."""
    inf = math.inf
    cost = torch.where(free, cost, inf)
    cost_p = F.pad(cost, (1, 1, 1, 1), value=inf)
    free_p = F.pad(free, (1, 1, 1, 1), value=False)
    weights = torch.zeros(*cost.shape, 3, 3, device=cost.device)
    for dx, dy in NEIGHBOURS:
        w = math.hypot(dx, dy) * 0.5 * (cost + _view(cost_p, dx, dy))
        if dx and dy:  # no corner cutting
            ok = _view(free_p, dx, 0) & _view(free_p, 0, dy)
            w = torch.where(ok, w, torch.full_like(w, inf))
        weights[..., 1 - dx, 1 - dy] = w
    return weights


def distances(weights: torch.Tensor, source: torch.Tensor, check_every: int = 4) -> torch.Tensor:
    """Bellman-Ford from one cell per map."""
    b, s = weights.shape[:2]
    rows = torch.arange(b, device=source.device)
    dist = torch.full((b, s, s), math.inf, device=source.device)
    dist[rows, source[:, 0], source[:, 1]] = 0.0
    for _ in range(s * s // check_every):
        before = dist
        for _ in range(check_every):
            windows = F.pad(dist, (1, 1, 1, 1), value=math.inf).unfold(1, 3, 1).unfold(2, 3, 1)
            dist = (windows + weights).amin(dim=(-1, -2))
        if torch.equal(before, dist):
            break
    return dist


def distances_from_field(weights: torch.Tensor, dist: torch.Tensor, check_every: int = 4):
    """Bellman-Ford from an initial ``(B, S, S)`` distance field (inf where unknown)."""
    s = weights.shape[1]
    for _ in range(s * s // check_every):
        before = dist
        for _ in range(check_every):
            windows = F.pad(dist, (1, 1, 1, 1), value=math.inf).unfold(1, 3, 1).unfold(2, 3, 1)
            dist = (windows + weights).amin(dim=(-1, -2))
        if torch.equal(before, dist):
            break
    return dist


def _open_endpoints(free, start, goal):
    free = free.clone()
    rows = torch.arange(free.shape[0], device=free.device)
    free[rows, start[:, 0], start[:, 1]] = True
    free[rows, goal[:, 0], goal[:, 1]] = True
    return free


@torch.no_grad()
def route_mask(cost, free, start, goal) -> torch.Tensor:
    """Cells on a cheapest start-goal route (``d_start + d_goal == d_start(goal)``)."""
    free = _open_endpoints(free, start, goal)
    rows = torch.arange(cost.shape[0], device=cost.device)
    weights = edge_weights(cost.float(), free)
    d_start, d_goal = distances(torch.cat([weights, weights]), torch.cat([start, goal])).chunk(2)
    total = d_start[rows, goal[:, 0], goal[:, 1]][:, None, None]
    return torch.isfinite(total) & (d_start + d_goal <= total * (1.0 + 1e-4) + 1e-4)


@torch.no_grad()
def route_paths(cost, free, start, goal) -> list[np.ndarray | None]:
    """One cheapest route per map as ``(L, 2)`` cell indices, None where unreachable."""
    free = _open_endpoints(free, start, goal)
    weights = edge_weights(cost.float(), free)
    dist = distances(weights, start).cpu().numpy()
    w = weights.cpu().numpy()
    s = dist.shape[-1]
    paths = []
    for k in range(dist.shape[0]):
        cur = (int(goal[k, 0]), int(goal[k, 1]))
        st = (int(start[k, 0]), int(start[k, 1]))
        if not math.isfinite(dist[k, cur[0], cur[1]]):
            paths.append(None)
            continue
        path = [cur]
        while cur != st:
            best, nxt = math.inf, None
            for dx, dy in NEIGHBOURS:
                px, py = cur[0] - dx, cur[1] - dy
                if 0 <= px < s and 0 <= py < s:
                    v = dist[k, px, py] + w[k, cur[0], cur[1], 1 - dx, 1 - dy]
                    if v < best:
                        best, nxt = v, (px, py)
            cur = nxt
            path.append(cur)
        paths.append(np.array(path[::-1], np.float32))
    return paths


def ridge_cost(h: torch.Tensor) -> torch.Tensor:
    """``1 - log max(h, TAU)``: unit length plus squared distance to the ridge in sigma units."""
    return 1.0 - torch.log(h.float().clamp(TAU, 1.0))


def projection_cost(h: torch.Tensor, theta: float = ON_RIDGE, eps: float = 1e-3) -> torch.Tensor:
    """``(theta - h)_+^2``: what lifting a cell to ``theta`` costs in L2; ``eps`` breaks ties
    towards shorter routes."""
    return (theta - h.float()).clamp_min(0.0).square() + eps


def bottleneck(h: torch.Tensor, free: torch.Tensor, start, goal) -> torch.Tensor:
    """Widest-path value start->goal, 8-connected without corner cutting."""
    rows = torch.arange(h.shape[0], device=h.device)
    hf = h.clamp(0, 1) * free
    free_p = F.pad(free, (1, 1, 1, 1), value=False)
    allowed = [
        (_view(free_p, dx, 0) & _view(free_p, 0, dy)).float() if dx and dy else None
        for dx, dy in NEIGHBOURS
    ]
    x = torch.zeros_like(hf)
    x[rows, start[:, 0], start[:, 1]] = hf[rows, start[:, 0], start[:, 1]]
    for _ in range(h.shape[-1] ** 2):
        x_p = F.pad(x, (1, 1, 1, 1))
        best = x
        for (dx, dy), ok in zip(NEIGHBOURS, allowed, strict=True):
            v = _view(x_p, dx, dy)
            best = torch.maximum(best, v if ok is None else v * ok)
        grown = torch.minimum(best, hf)
        if torch.equal(grown, x):
            break
        x = grown
    return x[rows, goal[:, 0], goal[:, 1]]


CORRIDOR_PX = 3  # 2 sigma of the ridge: beyond this a parallel ridge is a separate route


def second_route_free(main, free, start, goal, radius: int = CORRIDOR_PX):
    """Free cells for a *second* start-goal route: everything outside a ``radius`` corridor
    around ``main`` (a ``(B, S, S)`` route mask), plus discs of ``radius + 1`` at the endpoints
    so the second route can leave them. Returns ``(free2, corridor)``."""
    k = 2 * radius + 1
    corridor = F.max_pool2d(main.float()[:, None], k, 1, radius)[:, 0] > 0.5
    seeds = torch.zeros_like(free)
    rows = torch.arange(free.shape[0], device=free.device)
    seeds[rows, start[:, 0], start[:, 1]] = True
    seeds[rows, goal[:, 0], goal[:, 1]] = True
    discs = F.max_pool2d(seeds.float()[:, None], k + 2, 1, radius + 1)[:, 0] > 0.5
    return free & (~corridor | discs), corridor


@torch.no_grad()
def hedging(h, free, start, goal) -> tuple[torch.Tensor, torch.Tensor]:
    """``(b2, second)``: the bottleneck of the best start-goal route that stays out of the main
    route's corridor, and that route's off-corridor cells. ``b2 >= ON_RIDGE`` means the sample
    paints a second, separate, unbroken route (hedging)."""
    main = route_mask(ridge_cost(h), free, start, goal)
    free2, corridor = second_route_free(main, free, start, goal)
    b2 = bottleneck(h, free2, start, goal)
    second = route_mask(ridge_cost(h), free2, start, goal) & ~corridor
    return b2, second


@torch.no_grad()
def project(h: torch.Tensor, free, start, goal, theta: float = ON_RIDGE):
    """L2 projection of ``h`` onto {H : some free start-goal route has H >= theta everywhere}:
    lift the cells of the cheapest route under ``(theta - h)_+^2`` to ``theta``. Already
    connected heatmaps come back unchanged (their cheapest route costs nothing), so no check is
    needed first. Returns ``(projected, route)``."""
    route = route_mask(projection_cost(h, theta), free, start, goal)
    return torch.where(route, h.clamp_min(theta), h), route
