"""Diagnose ridge topology failures and compare cheap, inference-only repairs.

Every heatmap/grid here is indexed ``[x, y]`` like the walker's inputs.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack
from ridgeflow.model.targets import RIDGE_SIGMA, path_heatmap

TAU = 1e-3  # the walker's truncation: below this a ridge is "nothing"
_NB = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _shift(x: torch.Tensor, dx: int, dy: int, fill: float) -> torch.Tensor:
    """``out[i, j] = x[i - dx, j - dy]`` with ``fill`` off-grid."""
    out = torch.full_like(x, fill)
    s = x.shape[-1]
    out[:, max(dx, 0) : s + min(dx, 0), max(dy, 0) : s + min(dy, 0)] = x[
        :, max(-dx, 0) : s + min(-dx, 0), max(-dy, 0) : s + min(-dy, 0)
    ]
    return out


def bottleneck(h: torch.Tensor, free: torch.Tensor, start: torch.Tensor, goal: torch.Tensor):
    """Widest-path value start->goal through free cells: ``max_path min_v H(v)``.

    Grey-level reconstruction by dilation; differentiable, the gradient lands on the saddle.
    It is also the level at which start and goal merge in the superlevel-set filtration of H,
    i.e. the 0-dim persistence of the start-goal connection.
    """
    rows = torch.arange(h.shape[0], device=h.device)
    hf = h.clamp(0, 1) * free
    x = torch.zeros_like(hf)
    x[rows, start[:, 0], start[:, 1]] = hf[rows, start[:, 0], start[:, 1]]
    for _ in range(h.shape[-1] ** 2):
        grown = torch.minimum(F.max_pool2d(x[:, None], 3, 1, 1)[:, 0], hf)
        if torch.equal(grown, x):
            break
        x = grown
    return x[rows, goal[:, 0], goal[:, 1]]


def geodesic(h: torch.Tensor, free: torch.Tensor, start: torch.Tensor, goal: torch.Tensor):
    """Batched shortest path under ``cost = 1 - log max(H, TAU)`` per unit length.

    ``-log`` of a Gaussian ridge is ``d^2 / 2 sigma^2``, the squared distance to its centreline,
    so this is the most likely free route given the ridge; TAU bounds the price of a gap.
    No corner cutting. Returns a list of ``(L, 2)`` pixel paths, None where unreachable.
    """
    b, s, _ = h.shape
    dev = h.device
    rows = torch.arange(b, device=dev)
    cost = torch.where(free, 1.0 - torch.log(h.clamp(TAU, 1.0)), torch.full_like(h, math.inf))
    dist = torch.full_like(h, math.inf)
    dist[rows, start[:, 0], start[:, 1]] = 0.0
    weights = []
    for dx, dy in _NB:
        w = math.hypot(dx, dy) * 0.5 * (cost + _shift(cost, dx, dy, math.inf))
        if dx and dy:  # forbid cutting a blocked corner
            w = torch.where(
                _shift(free, dx, 0, False) & _shift(free, 0, dy, False),
                w,
                torch.full_like(w, math.inf),
            )
        weights.append(w)
    for _ in range(s * s):
        best = dist
        for (dx, dy), w in zip(_NB, weights, strict=True):
            best = torch.minimum(best, _shift(dist, dx, dy, math.inf) + w)
        if torch.equal(best, dist):
            break
        dist = best

    paths = []
    dist_c, w_c = dist.cpu(), [w.cpu() for w in weights]
    for k in range(b):
        g = (int(goal[k, 0]), int(goal[k, 1]))
        if not math.isfinite(float(dist_c[k, g[0], g[1]])):
            paths.append(None)
            continue
        path, cur, st = [g], g, (int(start[k, 0]), int(start[k, 1]))
        while cur != st:
            options = []
            for (dx, dy), w in zip(_NB, w_c, strict=True):
                px, py = cur[0] - dx, cur[1] - dy
                if 0 <= px < s and 0 <= py < s:
                    options.append((float(dist_c[k, px, py] + w[k, cur[0], cur[1]]), (px, py)))
            cur = min(options)[1]
            path.append(cur)
        paths.append(np.array(path[::-1], np.float32))
    return paths


def repaint(paths, size: int, device) -> torch.Tensor:
    """Clean training-style ridges along pixel paths, ``[x, y]``."""
    out = torch.zeros(len(paths), size, size, device=device)
    for k, p in enumerate(paths):
        if p is not None:
            out[k] = path_heatmap(p, size, RIDGE_SIGMA, device=device)[0].T
    return out


@torch.no_grad()
def projected_sample(model: RidgeModel, occ, starts, goals, free, si, gi, seed: int):
    """The usual 4-step sampler, but after each step's x0 estimate the ridge geodesic is painted
    in (``x0 <- max(x0, ridge(geodesic(x0)))``) and the remaining steps refine it."""
    flow, net, dev = model.flow, model.model, model.device
    obstacle = occ.to(dev, torch.float32).transpose(1, 2)[:, None]
    start, goal = model._endpoints(starts), model._endpoints(goals)
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, *obstacle.shape[-2:], device=dev)
    steps, eta = model.steps, model.eta
    schedule = [1.0 - i / steps for i in range(steps + 1)]
    for i in range(steps):
        t, t_next = schedule[i], schedule[i + 1]
        tt = torch.full((x.shape[0],), t * flow.time_scale, device=dev)
        v = net(x, obstacle, start, goal, tt)
        x0 = x - t * v
        h = x0[:, 0].transpose(1, 2).clamp(0, 1)  # [x, y]
        ridge = repaint(geodesic(h, free, si, gi), h.shape[-1], dev)
        x0 = torch.maximum(x0, ridge.transpose(1, 2)[:, None])
        if t_next > 0.0:
            noise = torch.randn_like(x)
            det = (1.0 - t_next) * x0 + t_next * (x - (1.0 - t) * x0) / t  # same eps, new t
            x = (1.0 - eta) * det + eta * ((1.0 - t_next) * x0 + t_next * noise)
        else:
            x = x0
    return x.clamp(0, 1)[:, 0].transpose(1, 2).contiguous()


def solved(worlds, result) -> np.ndarray:
    return np.array(
        [
            bool(ok) and w.path_free(p)
            for w, p, ok in zip(worlds, result.paths, result.reached, strict=True)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ridgeflow_rrt64.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=8, help="samples per query for best-of-K")
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid, device=args.device)
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(seed), grid) for seed in range(args.queries)]

    rec: dict[str, list] = {}

    def add(name, values):
        rec.setdefault(name, []).extend(np.asarray(values).tolist())

    began = time.perf_counter()
    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        free_pad, free_raw = ~grids, ~occ
        si = torch.floor(starts / grid.cell).long()
        gi = torch.floor(goals / grid.cell).long()
        n = len(chunk)

        # K samples per query; sample 0 is exactly what benchmark.py draws
        samples = [model.sample(occ, starts, goals, seed=begin)]
        for j in range(1, args.k):
            samples.append(model.sample(occ, starts, goals, seed=10_000 * j + begin))
        b_pad = torch.stack([bottleneck(h, free_pad, si, gi) for h in samples], 1)  # (n, K)
        b_raw = bottleneck(samples[0], free_raw, si, gi)
        h0 = samples[0]

        base = solved(chunk, walker.walk(h0, grids, starts, goals))
        add("base", base)
        add("b_pad", b_pad[:, 0].cpu())
        add("b_raw", b_raw.cpu())
        # mass of the ridge inside obstacles / inside the 1-px padding only
        mass = h0.clamp(0, 1).sum((1, 2))
        add("in_wall", ((h0.clamp(0, 1) * occ).sum((1, 2)) / mass).cpu())
        add("in_pad", ((h0.clamp(0, 1) * (grids & ~occ)).sum((1, 2)) / mass).cpu())

        # best-of-K by bottleneck (no threshold: argmax)
        for k in sorted({2, 4, args.k}):
            pick = b_pad[:, :k].argmax(1)
            hk = torch.stack([samples[int(p)][i] for i, p in enumerate(pick)])
            add(f"best_of_{k}", solved(chunk, walker.walk(hk, grids, starts, goals)))
        # oracle: any of K walks succeeds
        any_ok = np.zeros(n, bool)
        for h in samples:
            any_ok |= solved(chunk, walker.walk(h, grids, starts, goals))
        add(f"oracle_any_of_{args.k}", any_ok)

        # ridge geodesic on sample 0: follow it directly, or repaint it and walk
        paths = geodesic(h0, free_pad, si, gi)
        direct = []
        for w, p, s_, g_ in zip(
            chunk, paths, starts.cpu().numpy(), goals.cpu().numpy(), strict=True
        ):
            if p is None:
                direct.append(False)
                continue
            pts = np.concatenate([s_[None], (p[1:-1] + 0.5) * grid.cell, g_[None]])
            direct.append(w.path_free(pts))
        add("geodesic_direct", direct)
        clean = repaint(paths, grid.size, model.device)
        add("geodesic_repaint_walk", solved(chunk, walker.walk(clean, grids, starts, goals)))
        # only repair when the walker failed (cost paid on failures only)
        add("walk_then_repair", base | np.array(direct))
        # topology projection inside the sampler, then the plain walker
        hp = projected_sample(model, occ, starts, goals, free_pad, si, gi, seed=begin)
        add("projected_sampling", solved(chunk, walker.walk(hp, grids, starts, goals)))
        add("b_proj", bottleneck(hp, free_pad, si, gi).cpu())
        print(f"{begin + n}/{len(worlds)}  {time.perf_counter() - began:.0f}s", flush=True)

    base = np.array(rec["base"])
    bp, br = np.array(rec["b_pad"]), np.array(rec["b_raw"])
    print(f"\n{len(base)} random maps\n")
    for name, v in rec.items():
        if name.startswith(("base", "best", "oracle", "geodesic", "walk_", "projected")):
            print(f"{name:<24}{100 * np.mean(v):6.1f}%")
    fail = ~base
    print(f"\nwalker failures: {fail.sum()}")
    for thr in (0.05, 0.2, 0.5):
        print(
            f"  bottleneck(padded) < {thr}: {np.mean(bp[fail] < thr) * 100:5.1f}% of failures, "
            f"{np.mean(bp[base] < thr) * 100:5.1f}% of successes"
        )
    print(
        f"  broken on padded grid but connected on raw (<0.2 vs >=0.5, wall-hug): "
        f"{np.sum(fail & (bp < 0.2) & (br >= 0.5))}"
    )
    q = [0.1, 0.25, 0.5]
    print(
        f"  bottleneck quantiles {q}: fail {np.quantile(bp[fail], q).round(3)}, "
        f"ok {np.quantile(bp[base], q).round(3)}"
    )
    print(
        f"  bottleneck median: base {np.median(bp):.3f}, projected {np.median(rec['b_proj']):.3f}"
    )
    iw, ip = np.array(rec["in_wall"]), np.array(rec["in_pad"])
    print(
        f"  ridge mass in walls: fail {iw[fail].mean():.4f} ok {iw[base].mean():.4f}; "
        f"in padding: fail {ip[fail].mean():.4f} ok {ip[base].mean():.4f}"
    )


if __name__ == "__main__":
    main()
