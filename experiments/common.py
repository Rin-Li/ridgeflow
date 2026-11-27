"""Helpers shared by the experiment scripts: sampler variants, TMP64 queries and search, route
decoding, metrics and plotting. Run the scripts as modules from the repository root, e.g.
``python -m experiments.rrt_ablation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ridgeflow import RidgeModel
from ridgeflow.routes import ON_RIDGE, bottleneck, project, ridge_cost, route_paths
from ridgeflow.search import astar, focal

plt.rcParams["axes.unicode_minus"] = False

RRT_CHECKPOINT = "checkpoints/ridgeflow_rrt64.pt"
TMP64_CHECKPOINT = "outputs/runs/long_scratch/best.pt"
TMP64_DATA = "outputs/data/tmp64_test.npy"


# --------------------------------------------------------------------------------------------
# sampler variants


@dataclass(frozen=True)
class Variant:
    """The shipped 4-step sampler, plus per step i (t_i = 1 - i/4):

    * projection, if i is in ``project_at``: when the ridge of ``x0_hat`` is broken
      (b* < theta_i), lift the cells of its cheapest route under (theta_i - H)_+^2 to theta_i;
    * self-recurrence: run the step ``recur`` times, re-noising the new state back to t_i
      between repeats with the rectified-flow forward kernel
      x_t = a x_t' + sqrt(t^2 - a^2 t'^2) xi,  a = (1 - t) / (1 - t').
    """

    name: str
    project_at: tuple[int, ...] = ()
    thetas: tuple[float, ...] = (ON_RIDGE,) * 4
    recur: int = 1
    note: str = ""


ALL = (0, 1, 2, 3)
RAMP = (0.2, 0.3, 0.4, 0.5)
VARIANTS = [
    Variant("plain", note="no guidance"),
    Variant("last", (3,), note="project at last step only"),
    Variant("last2", (2, 3), note="project at last two steps"),
    Variant("all", ALL, note="project every step (θ=0.5)"),
    Variant("ramp", ALL, RAMP, note="project every step, θ: 0.2→0.5"),
    Variant("plain_r2", recur=2, note="no guidance + 2 repeats per step"),
    Variant("all_r2", ALL, recur=2, note="project every step + 2 repeats"),
    Variant("ramp_r2", ALL, RAMP, recur=2, note="rising θ + 2 repeats"),
]
BY_NAME = {v.name: v for v in VARIANTS}
PLAIN = BY_NAME["plain"]


@torch.no_grad()
def sample(model: RidgeModel, occ, grids, starts, goals, seed: int, v: Variant, trace=False):
    """``(B, S, S)`` [x, y] heatmaps under variant ``v`` (``grids`` is only read to project);
    also how many maps each step projected and, with ``trace``, per step the ``x0_hat`` before
    projection and the lifted cells. For the plain sampler call ``model.sample`` instead."""
    flow, net, dev = model.flow, model.model, model.device
    obstacle = occ.to(dev, torch.float32).transpose(1, 2)[:, None]
    start_h, goal_h = model._endpoints(starts), model._endpoints(goals)
    si, gi = model.grid.cells(starts), model.grid.cells(goals)
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, *obstacle.shape[-2:], device=dev)
    steps, eta = model.steps, model.eta
    schedule = [1.0 - i / steps for i in range(steps + 1)]
    frames: list = []
    projected = [0] * steps
    for i in range(steps):
        t, t_next = schedule[i], schedule[i + 1]
        tt = torch.full((x.shape[0],), t * flow.time_scale, device=dev)
        for r in range(v.recur):
            last = r == v.recur - 1
            vel = net(x, obstacle, start_h, goal_h, tt)
            x0 = x - t * vel
            h = x0[:, 0].transpose(1, 2)  # [x, y]
            before = h.clone() if trace and last else None
            if i in v.project_at:
                theta = v.thetas[i]
                rows = (bottleneck(h, ~grids, si, gi) < theta).nonzero()[:, 0]
                if len(rows):
                    h[rows] = project(h[rows], ~grids[rows], si[rows], gi[rows], theta)[0]
                if last:
                    projected[i] = len(rows)
            if before is not None:
                frames.append((before.cpu().numpy(), (h - before).abs().cpu().numpy() > 1e-6))
            x0 = h.transpose(1, 2)[:, None]
            if t_next > 0.0:
                noise = torch.randn_like(x)
                det = (1.0 - t_next) * x0 + t_next * (x - (1.0 - t) * x0) / t
                nxt = (1.0 - eta) * det + eta * ((1.0 - t_next) * x0 + t_next * noise)
            else:
                nxt = x0
            if not last:  # re-noise back to t and redo the step
                a = (1.0 - t) / (1.0 - t_next)
                std = max(t**2 - (a * t_next) ** 2, 0.0) ** 0.5
                x = a * nxt + std * torch.randn_like(x)
            else:
                x = nxt
    return x.clamp(0, 1)[:, 0].transpose(1, 2).contiguous(), projected, frames


# --------------------------------------------------------------------------------------------
# TMP64


def tmp64_queries(data: dict, n: int | None = None):
    """The first ``n`` (all) TMP64 queries as A*/FOCAL inputs: blocked grids, start and goal
    cells."""
    n = len(data["start"]) if not n else n
    blocked = [data["map"][data["map_id"][i]].astype(bool) for i in range(n)]
    starts = [tuple(c) for c in data["start_cell"][:n]]
    goals = [tuple(c) for c in data["goal_cell"][:n]]
    return blocked, starts, goals


def astar_baseline(pool, blocked, starts, goals) -> tuple[np.ndarray, np.ndarray]:
    """A* expansions and costs of every query."""
    jobs = [(b, s, g, None, 1.0) for b, s, g in zip(blocked, starts, goals, strict=True)]
    rows = pool.map(search, jobs)
    return np.array([e for e, _ in rows], float), np.array([c for _, c in rows])


def search(args):
    """``(blocked, start, goal, guide, w)`` -> (expanded, cost): A* when ``guide`` is None,
    else FOCAL with weight ``w`` guided by the heatmap. One tuple so it maps over a Pool."""
    blocked, start, goal, guide, w = args
    r = astar(blocked, start, goal) if guide is None else focal(blocked, start, goal, guide, w)
    return r.expanded, r.cost


def tmp64_heatmaps(model, data, idx, seed, variant, draws):
    """``draws`` samples for each TMP64 query in ``idx``, in one batch per 256 rows.

    Returns heatmaps ``(Q, draws, S, S)``, their ridge connectivity b* ``(Q, draws)`` and the
    same with the walls ignored ``(Q, draws)`` (does the ridge break, or only meet a wall?)."""
    occ = torch.from_numpy(data["map"][data["map_id"][idx]]).bool().to(model.device)
    starts = torch.from_numpy(data["start"][idx]).to(model.device)
    goals = torch.from_numpy(data["goal"][idx]).to(model.device)
    si = torch.from_numpy(data["start_cell"][idx]).to(model.device)
    gi = torch.from_numpy(data["goal_cell"][idx]).to(model.device)
    q, s = len(idx), model.grid.size

    def rep(t):
        return t.repeat_interleave(draws, 0)

    occ_r, st_r, go_r, si_r, gi_r = rep(occ), rep(starts), rep(goals), rep(si), rep(gi)
    hs, bs, walls = [], [], []
    for c in range(0, len(occ_r), 256):
        sl = slice(c, c + 256)
        h, _, _ = sample(model, occ_r[sl], occ_r[sl], st_r[sl], go_r[sl], seed + c, variant)
        hs.append(h)
        bs.append(bottleneck(h, ~occ_r[sl], si_r[sl], gi_r[sl]))
        walls.append(bottleneck(h, torch.ones_like(occ_r[sl]), si_r[sl], gi_r[sl]))
    return (
        torch.cat(hs).view(q, draws, s, s),
        torch.cat(bs).view(q, draws),
        torch.cat(walls).view(q, draws),
    )


# --------------------------------------------------------------------------------------------
# routes


def geodesic_paths(h, grids, starts, goals, grid) -> list[np.ndarray | None]:
    """Each map's ridge geodesic (cheapest route under ``1 - log H``) as a world path from the
    exact start to the exact goal through cell centres; None where the goal is unreachable."""
    cells = route_paths(ridge_cost(h), ~grids, grid.cells(starts), grid.cells(goals))
    out = []
    for c, s, g in zip(cells, starts.cpu().numpy(), goals.cpu().numpy(), strict=True):
        out.append(
            None if c is None else np.concatenate([s[None], (c[1:-1] + 0.5) * grid.cell, g[None]])
        )
    return out


def path_length(path) -> float:
    """Length of a polyline; NaN for None."""
    if path is None:
        return float("nan")
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


# --------------------------------------------------------------------------------------------
# metrics


def diversity(masks: np.ndarray) -> float:
    """Mean pairwise 1 - IoU of ``(K, S, S)`` boolean ridge masks."""
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = (masks[i] | masks[j]).sum()
            vals.append(1.0 - (masks[i] & masks[j]).sum() / max(union, 1))
    return float(np.mean(vals))


def turning(path: np.ndarray) -> tuple[float, float]:
    """Total and largest heading change (rad) along a polyline."""
    d = np.diff(path, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    if len(d) < 2:
        return 0.0, 0.0
    a = np.arctan2(d[:, 1], d[:, 0])
    t = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(t.sum()), float(t.max())


# --------------------------------------------------------------------------------------------
# plotting


def base_axes(ax, world, image, cmap="magma", vmax=1.0):
    """A world's heatmap (or any [x, y] image) with its obstacles, start and goal."""
    e = world.grid.extent
    ax.imshow(image.T, cmap=cmap, origin="lower", extent=[0, e, 0, e], vmin=0, vmax=vmax)
    occ = world.occupancy().numpy().T
    ax.imshow(
        np.ma.masked_where(~occ, occ),
        cmap="Greys",
        origin="lower",
        extent=[0, e, 0, e],
        vmin=0,
        vmax=1.4,
        alpha=0.95,
    )
    ax.plot(*world.start, "o", color="white", ms=7, mec="k", zorder=7)
    ax.plot(*world.goal, "*", color="cyan", ms=13, mec="k", zorder=7)
    ax.set_xlim(0, e)
    ax.set_ylim(0, e)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_search(ax, blocked, start, goal, h, closed, path, title):
    """A TMP64 query: heatmap (orange), cells FOCAL expanded (blue), walls and the path."""
    img = np.clip(h, 0, 1)[..., None] * np.array([1.0, 0.55, 0.1])
    for cx, cy in closed:
        img[cx, cy] = 0.55 * img[cx, cy] + 0.45 * np.array([0.35, 0.7, 1.0])
    img[blocked] = 0.4
    ax.imshow(img.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "-", color="#00e68a", lw=1.6)
    ax.plot(*start, "o", color="white", mec="k", ms=6)
    ax.plot(*goal, "*", color="cyan", mec="k", ms=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
