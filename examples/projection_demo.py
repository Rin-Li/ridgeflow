"""What the last-step projection does, shown on RRT benchmark worlds, and what it costs.

The projection lifts the cells of one start-goal route to ON_RIDGE (0.5) so the heatmap becomes
connected, changing H as little as possible in L2. The route it lifts is the cheapest one under
the cell cost (0.5 - H)_+^2 + eps: a cell already on the ridge costs only eps, a cell in a gap
costs what filling it in costs. So it is a shortest-path search, but on a cost map made from the
heatmap, not on the occupancy grid: on a connected heatmap it returns a route along the ridge
and changes nothing; on a broken one it bridges the gap with the fewest, cheapest cells.

Pages: the idea and timings, then example worlds (raw heatmap, cost map, distance valley,
projected heatmap against the plain geometric shortest path).
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from sampler_ablation import VARIANTS, sample  # noqa: E402
from topology_report import base_axes  # noqa: E402  (registers the CJK font)

from ridgeflow import GridSpec, RidgeModel, sample_rect_world, stack  # noqa: E402
from ridgeflow.routes import (  # noqa: E402
    ON_RIDGE,
    _open_endpoints,
    bottleneck,
    distances,
    edge_weights,
    project,
    projection_cost,
    route_paths,
)
from ridgeflow.search import astar  # noqa: E402

PLAIN = next(v for v in VARIANTS if v.name == "plain")


def bellman_ford_sweeps(cost, free, start):
    """How many 3x3 min-plus sweeps ``distances`` needs before nothing changes."""
    weights = edge_weights(cost.float(), _open_endpoints(free, start, start))
    rows = torch.arange(cost.shape[0], device=cost.device)
    dist = torch.full(cost.shape, float("inf"), device=cost.device)
    dist[rows, start[:, 0], start[:, 1]] = 0.0
    for sweep in range(1, cost.shape[-1] ** 2):
        windows = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=float("inf"))
        new = (windows.unfold(1, 3, 1).unfold(2, 3, 1) + weights).amin(dim=(-1, -2))
        if torch.equal(new, dist):
            return sweep
        dist = new
    return sweep


def length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if p is not None else np.nan


def timed(fn, reps=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/runs/rrt_full/best.pt")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--output", default="outputs/projection_demo.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    cell = grid.cell
    model = RidgeModel(args.model, grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]

    # one draw per world, in the same batches as rrt_ablation (draw 0)
    H, B, ROUTE, P, GEO, LIFT, DMAX = [], [], [], [], [], [], []
    t_proj = t_gen = 0.0
    for begin in range(0, len(worlds), 250):
        chunk = worlds[begin : begin + 250]
        occ, grids, starts, goals = stack(chunk, model.device)
        si, gi = (torch.floor(x / cell).long() for x in (starts, goals))
        dt, (h, _, _) = timed(lambda: sample(model, occ, grids, starts, goals, begin, PLAIN))  # noqa: B023
        t_gen += dt
        dt, (hp, route) = timed(lambda: project(h, ~grids, si, gi))  # noqa: B023
        t_proj += dt
        H.append(h.cpu().numpy())
        B.append(bottleneck(h, ~grids, si, gi).cpu().numpy())
        ROUTE.append(route.cpu().numpy())
        LIFT.append(((hp - h) > 1e-6).sum((1, 2)).cpu().numpy())
        DMAX.append((hp - h).amax((1, 2)).cpu().numpy())
        P += route_paths(projection_cost(h), ~grids, si, gi)
        GEO += route_paths(torch.ones_like(h), ~grids, si, gi)
    H, B, ROUTE, LIFT, DMAX = (np.concatenate(x) for x in (H, B, ROUTE, LIFT, DMAX))
    n = len(worlds)
    reach = np.array([q is not None for q in GEO])  # a few worlds have no route on the padded grid
    broken = (B < ON_RIDGE) & reach
    ratio = np.array([length(p) / length(q) for p, q in zip(P, GEO, strict=True)])

    # timings: batch of 1 vs batch of 250, Bellman-Ford sweeps, and a CPU A* for comparison
    occ, grids, starts, goals = stack(worlds[:250], model.device)
    si, gi = (torch.floor(x / cell).long() for x in (starts, goals))
    h = torch.from_numpy(H[:250]).to(model.device)
    project(h[:1], ~grids[:1], si[:1], gi[:1])  # warm up
    one = np.mean(
        [
            timed(lambda: project(h[i : i + 1], ~grids[i : i + 1], si[i : i + 1], gi[i : i + 1]))[0]  # noqa: B023
            for i in range(50)
        ]
    )
    many = timed(lambda: project(h, ~grids, si, gi), reps=5)[0] / 250
    sweeps = [
        bellman_ford_sweeps(projection_cost(h[i : i + 1]), ~grids[i : i + 1], si[i : i + 1])
        for i in range(50)
    ]
    blocked = grids.cpu().numpy()
    sc, gc = si.cpu().numpy(), gi.cpu().numpy()
    t0 = time.perf_counter()
    for i in range(250):
        astar(blocked[i], sc[i], gc[i])
    t_astar = (time.perf_counter() - t0) / 250

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "Final-step projection: how it works and what it costs",
            ha="center",
            fontsize=17,
            weight="bold",
        )
        lines = [
            ("The problem", True),
            (
                "A sampled heatmap H can break somewhere (gaps with H<0.5), "
                "and a walker following the ridge gets stuck. "
                "Projection = find the connected heatmap closest to H (L2).",
                False,
            ),
            ("Method (one shortest path)", True),
            (
                "1. Per-cell cost c = (0.5 − H)₊² + ε (ε=1e-3): cells already on the ridge "
                "are nearly free, gap cells pay the L2 cost of lifting them to 0.5.",
                False,
            ),
            (
                "2. On GPU, run Bellman-Ford from the start and from the goal "
                "(repeated 3×3 min-plus convolutions, whole batch of maps at once) "
                "to get d_s, d_g.",
                False,
            ),
            (
                "3. Cells with d_s + d_g = d_s(goal) form the cheapest route; "
                "lift the H<0.5 cells on it to 0.5, and leave every other cell untouched.",
                False,
            ),
            ("How it differs from a plain shortest path", True),
            (
                "It is a shortest-path search, but the cost comes from the heatmap, not geometric "
                "distance: "
                "when the heatmap is connected, the route follows the ridge the model drew, "
                "and H is unchanged;",
                False,
            ),
            (
                "when broken, only the minimum number of cells is patched at the gaps, "
                "and the route still takes the corridor the model chose (homotopy class), "
                "not the geometric shortest path.",
                False,
            ),
            (
                "To be honest: it is a whole-map search and guarantees feasibility by itself, "
                "so after projection every model (even the old one) reaches "
                "walker success rate of 99.8–99.9%.",
                False,
            ),
            (
                "So on RRT the final success rate is mostly guaranteed by projection; "
                "the model's value shows in single-draw connectivity, "
                "how many cells get edited, and whether the route follows the model's ridge.",
                False,
            ),
        ]
        y = 0.88
        for s, head in lines:
            fig.text(0.04, y, s, fontsize=11 if head else 9.5, weight="bold" if head else None)
            y -= 0.038 if head else 0.033
        nb = int(broken.sum())
        cells = [
            ["Connected in one draw (no edits)", f"{100 * (reach & ~broken).mean():.1f}%"],
            ["Unsolvable maps (start/goal disconnected)", f"{int((~reach).sum())}/{n}"],
            ["Broken, needs patching", f"{100 * broken.mean():.1f}% ({nb}/{n})"],
            [
                "When broken: mean lifted cells / max lift",
                f"{LIFT[broken].mean():.1f} cells / {DMAX[broken].max():.3f}" if nb else "—",
            ],
            [
                "Tweaked even when connected (ε cuts corners)",
                f"{100 * (LIFT[reach & ~broken] > 0).mean():.1f}%, "
                f"max lift {DMAX[reach & ~broken].max():.3f}",
            ],
            [
                "Projected / geometric shortest path length (median, mean)",
                f"{np.nanmedian(ratio):.3f}, {np.nanmean(ratio):.3f}",
            ],
            [
                "Path >5% longer than geometric shortest (other corridor)",
                f"{100 * np.nanmean(ratio > 1.05):.1f}%",
            ],
            ["Generate one heatmap (4-step Euler, batch 250)", f"{1000 * t_gen / n:.2f} ms"],
            [
                "Projection (batch 250 amortized)",
                f"{1000 * t_proj / n:.2f} ms (re-measured {1000 * many:.2f} ms)",
            ],
            ["Projection (single map, batch 1)", f"{1000 * one:.2f} ms"],
            ["Bellman-Ford sweeps (median / max)", f"{int(np.median(sweeps))} / {max(sweeps)}"],
            ["Reference: Python A* on CPU, grid (single map)", f"{1000 * t_astar:.2f} ms"],
        ]
        ax = fig.add_axes([0.08, 0.02, 0.84, 0.42])
        ax.axis("off")
        tab = ax.table(
            cellText=cells, colLabels=["Item", "Value"], loc="upper center", cellLoc="left"
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        tab.scale(1, 1.45)
        pdf.savefig(fig)
        plt.close(fig)

        # examples: broken ones first, then connected ones whose route differs from geometry
        picks = list(np.flatnonzero(broken)[:5])  # broken already excludes unreachable worlds
        picks += list(np.flatnonzero(reach & ~broken & (ratio > 1.1))[:2])
        picks += list(np.flatnonzero(reach & ~broken & (ratio <= 1.1))[:1])
        for p0 in range(0, len(picks), 2):
            fig, axes = plt.subplots(2, 4, figsize=(11.69, 7.4))
            for row, q in zip(axes, picks[p0 : p0 + 2], strict=False):
                w = worlds[q]
                h = H[q]
                hp = np.where(ROUTE[q], np.maximum(h, ON_RIDGE), h)
                lifted = ROUTE[q] & (h < ON_RIDGE)
                occ1, grid1, s1, g1 = stack([w], model.device)
                sv, gv = (torch.floor(x / cell).long() for x in (s1, g1))
                cost = projection_cost(torch.from_numpy(h)[None].to(model.device))
                wts = edge_weights(cost.float(), _open_endpoints(~grid1, sv, gv))
                ds, dg = distances(torch.cat([wts, wts]), torch.cat([sv, gv])).cpu().numpy()
                valley = ds + dg
                valley = np.where(np.isfinite(valley), valley, np.nan)
                xy = lambda m: (np.argwhere(m) + 0.5) * cell  # noqa: E731
                base_axes(row[0], w, h)
                gaps = xy(lifted)
                row[0].plot(gaps[:, 0], gaps[:, 1], "s", color="red", ms=3.5, mfc="none")
                row[0].set_title(
                    f"#{q} heatmap H  b*={B[q]:.2f}\nred box = gap cells to patch", fontsize=8
                )
                c = np.where(grid1[0].cpu().numpy(), np.nan, cost[0].cpu().numpy())
                base_axes(row[1], w, np.zeros_like(h))
                im = row[1].imshow(
                    c.T,
                    origin="lower",
                    extent=[0, grid.extent] * 2,
                    cmap="viridis",
                    norm=LogNorm(1e-3, 0.26),
                )
                fig.colorbar(im, ax=row[1], fraction=0.046)
                row[1].set_title(
                    "per-cell cost (0.5−H)₊²+ε (log)\nridge ≈ ε, gaps expensive", fontsize=8
                )
                base_axes(row[2], w, np.zeros_like(h))
                total = np.nanmin(valley)
                im = row[2].imshow(
                    (valley - total).T,
                    origin="lower",
                    extent=[0, grid.extent] * 2,
                    cmap="magma_r",
                    vmin=0,
                    vmax=max(5 * total, 0.02),
                )
                fig.colorbar(im, ax=row[2], fraction=0.046)
                row[2].set_title(
                    f"d_s + d_g − min (valley = route)\nroute total cost {total:.3f}", fontsize=8
                )
                base_axes(row[3], w, hp)
                if lifted.any():
                    row[3].plot(gaps[:, 0], gaps[:, 1], "s", color="red", ms=3.5, mfc="none")
                for path, style, lab in (
                    (GEO[q], "--", "geometric shortest"),
                    (P[q], "-", "projected route"),
                ):
                    pts = (path + 0.5) * cell
                    row[3].plot(
                        pts[:, 0],
                        pts[:, 1],
                        style,
                        lw=1.4,
                        color="#00e5ff" if style == "--" else "#00e68a",
                        label=lab,
                    )
                row[3].legend(fontsize=6, loc="lower right")
                row[3].set_title(
                    f"H' after projection: patched {int(lifted.sum())} cells\n"
                    f"path length / geometric shortest = {ratio[q]:.2f}",
                    fontsize=8,
                )
                for a in row:
                    a.set_xticks([])
                    a.set_yticks([])
            for a in axes[len(picks[p0 : p0 + 2]) :].ravel():
                a.axis("off")
            fig.suptitle(
                "Examples: first rows are broken draws, later rows are connected "
                "draws that took another corridor",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)
    print(
        f"broken {broken.sum()}/{n}, lifted mean {LIFT[broken].mean() if broken.any() else 0:.1f}, "
        f"gen {1000 * t_gen / n:.2f} ms, proj {1000 * t_proj / n:.2f} / "
        f"single {1000 * one:.2f} ms, "
        f"sweeps median {np.median(sweeps)}, astar {1000 * t_astar:.2f} ms, "
        f"ratio median {np.nanmedian(ratio):.3f}, >5% {100 * np.nanmean(ratio > 1.05):.1f}%"
    )


if __name__ == "__main__":
    main()
