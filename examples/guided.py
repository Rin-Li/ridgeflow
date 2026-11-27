"""Connectivity-guided sampling x stage-2 decoders, with a results PDF.

Samplers (4 Euler steps, as shipped):
  plain     the shipped sampler
  proj_log  every step: paint the -log H geodesic into x0_hat (MAP route), always
  proj_l2   every step: if b*(x0_hat) < theta, replace x0_hat by its L2 projection onto
            {H : some free route has H >= theta everywhere}; that projection is exactly a
            cheapest route under (theta - H)_+^2 with its cells lifted to theta

Decoders (stage 2):
  walker      the ridge walker
  geodesic    cheapest free route under 1 - log max(H, tau)  (HeatPath's A* init, our cost)
  refine      geodesic -> 64 points -> Adam on render-MSE (HeatPath's refiner, endpoints fixed)
  refine_col  same, plus a clearance penalty on the padded grid's distance field

All heatmaps/grids here are [x, y].
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from connectivity_eval import diversity  # noqa: E402
from topology_report import base_axes  # noqa: E402  (registers the CJK font)

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402
from ridgeflow.model.targets import RIDGE_SIGMA, path_heatmap  # noqa: E402
from ridgeflow.routes import (  # noqa: E402
    ON_RIDGE,
    bottleneck,
    projection_cost,
    ridge_cost,
    route_mask,
    route_paths,
)

SAMPLERS = ("plain", "proj_log", "proj_l2")
DECODERS = ("walker", "geodesic", "refine", "refine_col")
RAW = ("refine_raw", "refine_col_raw")


# ----------------------------------------------------------------------------- sampling
def repaint(paths, size, device):
    out = torch.zeros(len(paths), size, size, device=device)
    for k, p in enumerate(paths):
        if p is not None:
            out[k] = path_heatmap(p, size, RIDGE_SIGMA, device=device)[0].T
    return out


@torch.no_grad()
def sample(model: RidgeModel, occ, grids, starts, goals, seed: int, mode: str, trace=False):
    """Returns final heatmaps ``(B, S, S)`` [x, y], per-step projected counts, optional trace."""
    flow, net, dev = model.flow, model.model, model.device
    cell = model.grid.cell
    obstacle = occ.to(dev, torch.float32).transpose(1, 2)[:, None]
    start_h, goal_h = model._endpoints(starts), model._endpoints(goals)
    si = torch.floor(starts / cell).long()
    gi = torch.floor(goals / cell).long()
    free = ~grids
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, *obstacle.shape[-2:], device=dev)
    steps, eta = model.steps, model.eta
    schedule = [1.0 - i / steps for i in range(steps + 1)]
    touched, frames = [], []
    for i in range(steps):
        t, t_next = schedule[i], schedule[i + 1]
        tt = torch.full((x.shape[0],), t * flow.time_scale, device=dev)
        v = net(x, obstacle, start_h, goal_h, tt)
        x0 = x - t * v
        h = x0[:, 0].transpose(1, 2)  # [x, y]
        before = h.clone()
        changed = torch.zeros(h.shape[0], dtype=torch.bool, device=dev)
        if mode == "proj_log":
            ridge = repaint(route_paths(ridge_cost(h.clamp(0, 1)), free, si, gi), h.shape[-1], dev)
            h = torch.maximum(h, ridge)
            changed[:] = True
        elif mode == "proj_l2":
            changed = bottleneck(h, free, si, gi) < ON_RIDGE
            if bool(changed.any()):
                rows = changed.nonzero()[:, 0]
                lift = route_mask(projection_cost(h[rows]), free[rows], si[rows], gi[rows])
                h[rows] = torch.where(lift, h[rows].clamp_min(ON_RIDGE), h[rows])
        touched.append(int(changed.sum()))
        if trace:
            frames.append((before.cpu().numpy(), (h - before).abs().cpu().numpy() > 1e-6))
        x0 = h.transpose(1, 2)[:, None]
        if t_next > 0.0:
            noise = torch.randn_like(x)
            det = (1.0 - t_next) * x0 + t_next * (x - (1.0 - t) * x0) / t
            x = (1.0 - eta) * det + eta * ((1.0 - t_next) * x0 + t_next * noise)
        else:
            x = x0
    return x.clamp(0, 1)[:, 0].transpose(1, 2).contiguous(), touched, frames


# ----------------------------------------------------------------------------- decoding
def _resample(points: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-8:
        return np.repeat(points[:1], n, 0)
    s = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])], -1)


def render(paths: torch.Tensor, size: int, sigma: float = RIDGE_SIGMA) -> torch.Tensor:
    """``(B, N, 2)`` pixel points [x, y] -> ``(B, S, S)`` [x, y] max-of-Gaussians (HeatPath)."""
    axis = torch.arange(size, device=paths.device, dtype=paths.dtype)
    dx2 = (axis[None, None, :, None] - paths[..., 0, None, None]).square()
    dy2 = (axis[None, None, None, :] - paths[..., 1, None, None]).square()
    return torch.exp(-(dx2 + dy2) / (2 * sigma**2)).amax(1)


def clearance(grids: torch.Tensor) -> torch.Tensor:
    """Distance (px) from each cell centre to the nearest padded-blocked cell centre."""
    return torch.from_numpy(
        np.stack([distance_transform_edt(~g) for g in grids.cpu().numpy()]).astype(np.float32)
    ).to(grids.device)


def refine(h, init_px, start_px, goal_px, field=None, iters=200, lr=0.03, n=64, margin=1.0):
    """Batched HeatPath refiner. ``init_px`` is a list of ``(L, 2)`` pixel paths [x, y]."""
    size = h.shape[-1]
    dev = h.device
    init = np.stack([_resample(p, n) for p in init_px]).astype(np.float32)
    params = torch.nn.Parameter(torch.from_numpy(init[:, 1:-1]).to(dev))
    s, g = start_px[:, None].float(), goal_px[:, None].float()
    opt = torch.optim.Adam([params], lr=lr)
    target = h.clamp(0, 1)
    with torch.enable_grad():
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            path = torch.cat([s, params, g], 1)
            loss = (render(path, size) - target).square().mean((1, 2)).sum()
            if field is not None:
                # clearance at the points and along each segment (quarter steps), bilinear;
                # grid_sample wants [-1, 1] coords with x -> W, y -> H
                frac = torch.tensor([0.0, 0.25, 0.5, 0.75], device=dev)[None, None, :, None]
                sub = path[:, :-1, None] + frac * (path[:, 1:, None] - path[:, :-1, None])
                coords = (sub.reshape(sub.shape[0], -1, 2) / (size - 1)) * 2 - 1
                d = F.grid_sample(
                    field[:, None].transpose(2, 3), coords[:, None], align_corners=True
                )[:, 0, 0]
                loss = loss + 10.0 * (margin - d).clamp_min(0).square().mean(1).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                params.clamp_(0, size - 1)
    return torch.cat([s, params.detach(), g], 1).cpu().numpy()


def decode_all(h, worlds, grids, starts, goals, walker, cell):
    """Success flags per decoder (padded grid; ``<name>_raw``: raw obstacles, as HeatPath
    checks), plus the paths (world coords) for figures."""
    from ridgeflow import World

    raw = [World(w.rects, w.start, w.goal, w.grid, padding_px=0) for w in worlds]
    dev = h.device
    si = torch.floor(starts / cell).long()
    gi = torch.floor(goals / cell).long()
    sp, gp = starts / cell - 0.5, goals / cell - 0.5
    out, paths, timing = {}, {}, {}

    t = time.perf_counter()
    res = walker.walk(h, grids, starts, goals)
    timing["walker"] = time.perf_counter() - t
    out["walker"] = [
        bool(r) and w.path_free(p) for w, p, r in zip(worlds, res.paths, res.reached, strict=True)
    ]
    paths["walker"] = res.paths

    t = time.perf_counter()
    cells = route_paths(ridge_cost(h), ~grids, si, gi)
    timing["geodesic"] = time.perf_counter() - t
    sp_np, gp_np = sp.cpu().numpy(), gp.cpu().numpy()
    init = []
    for k, c in enumerate(cells):
        if c is None:
            init.append(np.stack([sp_np[k], gp_np[k]]))
        else:
            init.append(np.concatenate([sp_np[k][None], c[1:-1], gp_np[k][None]]))
    geo_world = [(p + 0.5) * cell for p in init]
    out["geodesic"] = [
        c is not None and w.path_free(p) for w, p, c in zip(worlds, geo_world, cells, strict=True)
    ]
    paths["geodesic"] = geo_world

    field = clearance(grids)
    for name, fld in (("refine", None), ("refine_col", field)):
        t = time.perf_counter()
        ref = refine(h, init, sp.to(dev), gp.to(dev), fld)
        timing[name] = time.perf_counter() - t + timing["geodesic"]
        world_paths = [(p + 0.5) * cell for p in ref]
        out[name] = [w.path_free(p) for w, p in zip(worlds, world_paths, strict=True)]
        out[f"{name}_raw"] = [w.path_free(p) for w, p in zip(raw, world_paths, strict=True)]
        paths[name] = world_paths
    return out, paths, timing


# ----------------------------------------------------------------------------- figures
COLOURS = {"walker": "#00e68a", "geodesic": "#3fa9ff", "refine_col": "#ffd400"}


def panel(ax, world, h, paths, oks, title):
    base_axes(ax, world, h)
    for name, colour in COLOURS.items():
        p = paths[name]
        ax.plot(p[:, 0], p[:, 1], "-", color=colour, lw=1.4, zorder=5, alpha=0.9)
        if not oks[name]:
            ax.plot(
                *p[-1] if name == "walker" else p[len(p) // 2],
                "x",
                color=colour,
                ms=8,
                mew=2.5,
                zorder=6,
            )
    ax.set_title(title, fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoints", nargs="+", help="name=path")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=2, help="noise seeds per map")
    parser.add_argument("--output", default="outputs/guided_report.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    rows, keep = [], {}
    for item in args.checkpoints:
        mname, path = item.split("=", 1)
        model = RidgeModel(path, grid)
        occ, grids, starts, goals = stack(worlds[:8], model.device)
        sample(model, occ, grids, starts, goals, 0, "proj_l2")  # warm-up
        for mode in SAMPLERS:
            stat = {d: [] for d in DECODERS + RAW} | {"conn": [], "touched": [], "div": []}
            t_sample, t_dec = 0.0, {d: 0.0 for d in DECODERS}
            for begin in range(0, len(worlds), args.batch):
                chunk = worlds[begin : begin + args.batch]
                occ, grids, starts, goals = stack(chunk, model.device)
                si = torch.floor(starts / grid.cell).long()
                gi = torch.floor(goals / grid.cell).long()
                masks = []
                for j in range(args.k):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    h, touched, _ = sample(
                        model, occ, grids, starts, goals, begin + 10_000 * j, mode
                    )
                    torch.cuda.synchronize()
                    t_sample += time.perf_counter() - t0
                    stat["conn"] += (bottleneck(h, ~grids, si, gi) >= ON_RIDGE).tolist()
                    stat["touched"].append(touched)
                    ok, paths, timing = decode_all(
                        h, chunk, grids, starts, goals, walker, grid.cell
                    )
                    for d in DECODERS:
                        stat[d] += ok[d]
                        t_dec[d] += timing[d]
                    for d in RAW:
                        stat[d] += ok[d]
                    masks.append((h >= ON_RIDGE).cpu().numpy())
                    if j == 0:
                        hn = h.cpu().numpy()
                        for k in range(len(chunk)):
                            keep.setdefault((mname, begin + k), {})[mode] = (
                                hn[k],
                                {d: paths[d][k] for d in COLOURS},
                                {d: ok[d][k] for d in COLOURS},
                            )
                stat["div"] += [diversity(m) for m in np.stack(masks, 1)]
            n = len(worlds) * args.k
            touched = np.sum(stat["touched"], 0) / n
            row = {
                "model": mname,
                "sampler": mode,
                "conn": 100 * np.mean(stat["conn"]),
                **{d: 100 * np.mean(stat[d]) for d in DECODERS + RAW},
                "div": float(np.mean(stat["div"])),
                "t_sample": 1000 * t_sample / n,
                **{f"t_{d}": 1000 * t_dec[d] / n for d in DECODERS},
                "touched": " ".join(f"{100 * v:.0f}" for v in touched),
            }
            rows.append(row)
            print(row, flush=True)

    # --------------------------------------------------------------------------- PDF
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = args.checkpoints[0].split("=", 1)[0]
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "In-sampling connectivity guidance × stage-2 decoding",
            ha="center",
            fontsize=16,
            weight="bold",
        )
        intro = [
            f"{len(worlds)} benchmark maps × {args.k} noise seeds; success = reaches goal without "
            "touching the 1px padding; raw-obstacles column checks raw rectangles only (HeatPath).",
            "Samplers: plain = no guidance | "
            "proj_log = paint the −log H geodesic into x̂0 each step | "
            "proj_l2 = check b* each step, L2 projection only if broken",
            "  (projection cost (θ−H)₊², θ=0.5, gaps lifted to θ)",
            "Decoders: walker | geodesic (HeatPath A* init, cost 1−log max(H,τ)) | "
            "refine = geodesic + render-MSE refinement (HeatPath, 200 Adam steps) | "
            "refine_col = refine + wall penalty",
            "Per-step proj. %: share of samples projected by proj_l2 at steps 1-4.",
        ]
        for i, s in enumerate(intro):
            fig.text(0.04, 0.89 - 0.025 * i, s, fontsize=8.5)
        cols = [
            "Model",
            "Sampler",
            "Heatmap conn.",
            "walker",
            "Geodesic",
            "refine",
            "refine_col",
            "Diversity",
            "Sample ms",
            "Geodesic ms",
            "refine ms",
            "Per-step proj. %",
        ]
        cells = [
            [
                r["model"],
                r["sampler"],
                f"{r['conn']:.1f}%",
                f"{r['walker']:.1f}%",
                f"{r['geodesic']:.1f}%",
                f"{r['refine']:.1f}%",
                f"{r['refine_col']:.1f}%",
                f"{r['div']:.3f}",
                f"{r['t_sample']:.2f}",
                f"{r['t_geodesic']:.2f}",
                f"{r['t_refine_col']:.2f}",
                r["touched"] if r["sampler"] == "proj_l2" else "—",
            ]
            for r in rows
        ]
        ax = fig.add_axes([0.03, 0.08, 0.94, 0.66])
        ax.axis("off")
        tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.7)
        pdf.savefig(fig)
        plt.close(fig)

        # maps where the plain sample of the first model is broken
        broken = [
            s
            for s in range(len(worlds))
            if (first, s) in keep and not keep[(first, s)]["plain"][2]["walker"]
        ][:16]
        for p in range(0, len(broken), 4):
            fig, axes = plt.subplots(4, 3, figsize=(8.27, 11.0))
            fig.suptitle(
                f"{first}: maps where walker fails under plain ({p + 1}-{p + 4})  "
                "green=walker blue=geodesic yellow=refine_col  x=fail",
                fontsize=10,
                weight="bold",
            )
            for r, s in enumerate(broken[p : p + 4]):
                for c, mode in enumerate(SAMPLERS):
                    h, pth, oks = keep[(first, s)][mode]
                    flags = " ".join(f"{d[:3]}{'✓' if oks[d] else '✗'}" for d in COLOURS)
                    panel(axes[r, c], worlds[s], h, pth, oks, f"#{s} {mode}  {flags}")
            for ax in axes[len(broken[p : p + 4]) :].ravel():
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

        # per-step evolution of x0_hat, plain vs proj_l2, on two broken maps
        model = RidgeModel(args.checkpoints[0].split("=", 1)[1], grid)
        for s in broken[:3]:
            world = worlds[s]
            occ, grids, starts, goals = stack([world], model.device)
            fig, axes = plt.subplots(2, 5, figsize=(11.69, 5.2))
            fig.suptitle(
                f"map #{s}: x̂0 per step (top plain, bottom proj_l2; cyan = cells lifted)",
                fontsize=11,
                weight="bold",
            )
            for r, mode in enumerate(("plain", "proj_l2")):
                h, _, frames = sample(
                    model,
                    occ,
                    grids,
                    starts,
                    goals,
                    (s // args.batch) * args.batch,
                    mode,
                    trace=True,
                )
                for c, (x0, lifted) in enumerate(frames):
                    base_axes(axes[r, c], world, np.clip(x0[0], 0, 1))
                    xs, ys = np.nonzero(lifted[0])
                    axes[r, c].scatter(
                        (xs + 0.5) * grid.cell,
                        (ys + 0.5) * grid.cell,
                        s=4,
                        c="cyan",
                        marker="s",
                        lw=0,
                        zorder=6,
                    )
                    axes[r, c].set_title(f"{mode} step {c + 1} x̂0", fontsize=8)
                base_axes(axes[r, 4], world, h[0].cpu().numpy())
                axes[r, 4].set_title(f"{mode} final output", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
