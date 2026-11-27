"""Sampler ablation, no retraining: where to project, a rising threshold, and self-recurrence.

Every variant is the shipped 4-step sampler plus, per step i (t_i = 1 - i/4):
  * projection (if step i is in ``project_at``): when b*(x0_hat) < theta_i, replace x0_hat by
    its L2 projection onto {H : some free route has H >= theta_i}, i.e. lift the cells of the
    cheapest route under (theta_i - H)_+^2 to theta_i;
  * self-recurrence (Universal Guidance): run the step ``recur`` times; between repeats the new
    state is re-noised back to t_i with the rectified-flow forward kernel
    x_t = a x_t' + sqrt(t^2 - a^2 t'^2) xi,  a = (1 - t) / (1 - t').
Metrics: output connectivity, hedging (second unbroken route), walker / geodesic success,
diversity across seeds, network evaluations and time per map. Writes a results PDF.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.common import VARIANTS, Variant, base_axes, diversity, geodesic_paths, sample
from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack
from ridgeflow.routes import (
    ON_RIDGE,
    bottleneck,
    hedging,
    ridge_cost,
    route_mask,
)


def draw_route(ax, world, h, route, b, title):
    """``base_axes`` plus the cells of a route: on the ridge (blue) and its gaps (red)."""
    base_axes(ax, world, h)
    cell = world.grid.cell
    gaps = route & (h < ON_RIDGE)
    on = route & ~gaps
    for mask, colour, size in ((on, "#3fa9ff", 3), (gaps, "#ff3b3b", 9)):
        xs, ys = np.nonzero(mask)
        ax.scatter(
            (xs + 0.5) * cell, (ys + 0.5) * cell, s=size, c=colour, marker="s", lw=0, zorder=6
        )
    ax.set_title(f"{title}\nb*={b:.2f} gaps {int(gaps.sum())}", fontsize=8)


@dataclass
class Stats:
    conn: list = field(default_factory=list)
    hedge: list = field(default_factory=list)
    walker: list = field(default_factory=list)
    geo: list = field(default_factory=list)
    div: list = field(default_factory=list)
    projected: np.ndarray = field(default_factory=lambda: np.zeros(4))
    seconds: float = 0.0


def evaluate(model, worlds, walker, v: Variant, k: int, batch: int, keep: dict, mname: str):
    grid = model.grid
    st = Stats()
    for begin in range(0, len(worlds), batch):
        chunk = worlds[begin : begin + batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si, gi = grid.cells(starts), grid.cells(goals)
        masks = []
        for j in range(k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h, projected, _ = sample(model, occ, grids, starts, goals, begin + 10_000 * j, v)
            torch.cuda.synchronize()
            st.seconds += time.perf_counter() - t0
            st.projected += np.array(projected)
            st.conn += (bottleneck(h, ~grids, si, gi) >= ON_RIDGE).tolist()
            st.hedge += (hedging(h, ~grids, si, gi)[0] >= ON_RIDGE).tolist()
            res = walker.walk(h, grids, starts, goals)
            st.walker += [
                bool(r) and w.path_free(p)
                for w, p, r in zip(chunk, res.paths, res.reached, strict=True)
            ]
            st.geo += [
                p is not None and w.path_free(p)
                for w, p in zip(chunk, geodesic_paths(h, grids, starts, goals, grid), strict=True)
            ]
            masks.append((h >= ON_RIDGE).cpu().numpy())
            if j == 0:
                hn = h.cpu().numpy()
                for q in range(len(chunk)):
                    keep[(mname, v.name, begin + q)] = hn[q]
        st.div += [diversity(m) for m in np.stack(masks, 1)]
    return st


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoints", nargs="+", help="name=path")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output", default="outputs/sampler_ablation.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    rows, keep, models = [], {}, {}
    for item in args.checkpoints:
        mname, path = item.split("=", 1)
        model = models[mname] = RidgeModel(path, grid)
        occ, grids, starts, goals = stack(worlds[:8], model.device)
        sample(model, occ, grids, starts, goals, 0, VARIANTS[3])  # warm-up
        for v in VARIANTS:
            st = evaluate(model, worlds, walker, v, args.k, args.batch, keep, mname)
            n = len(worlds) * args.k
            row = dict(
                model=mname,
                variant=v.name,
                note=v.note,
                nfe=4 * v.recur,
                conn=100 * np.mean(st.conn),
                hedge=100 * np.mean(st.hedge),
                walker=100 * np.mean(st.walker),
                geo=100 * np.mean(st.geo),
                div=float(np.mean(st.div)),
                ms=1000 * st.seconds / n,
                proj=" ".join(f"{100 * p / n:.0f}" for p in st.projected) if v.project_at else "—",
            )
            rows.append(row)
            print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "Sampler ablation: projection timing × threshold × self-recurrence",
            ha="center",
            fontsize=15,
            weight="bold",
        )
        intro = [
            f"{len(worlds)} benchmark maps × {args.k} noise seeds."
            "Projection = if b*(x̂0) < θ, raise the cells on the shortest route under cost "
            "(θ−H)₊² to θ (minimal L2 change).",
            "Heatmap connectivity = output H has a route from start to goal with H ≥ 0.5 "
            "throughout; multi-route = after removing the main route's 3px corridor, "
            "another route with H ≥ 0.5 throughout remains (hedging).",
            "Per-step projection % = share of samples judged disconnected and projected at steps "
            "1-4. NFE = network calls.",
        ]
        for i, s in enumerate(intro):
            fig.text(0.04, 0.89 - 0.03 * i, s, fontsize=9.5)
        cols = [
            "model",
            "variant",
            "note",
            "NFE",
            "heatmap conn.",
            "multi-route",
            "walker",
            "geodesic",
            "diversity",
            "ms/map",
            "per-step proj. %",
        ]
        cells = [
            [
                r["model"],
                r["variant"],
                r["note"],
                r["nfe"],
                f"{r['conn']:.1f}%",
                f"{r['hedge']:.1f}%",
                f"{r['walker']:.1f}%",
                f"{r['geo']:.1f}%",
                f"{r['div']:.3f}",
                f"{r['ms']:.1f}",
                r["proj"],
            ]
            for r in rows
        ]
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.72])
        ax.axis("off")
        tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1, 1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # maps where plain is broken, for the first model
        first = args.checkpoints[0].split("=", 1)[0]
        model = models[first]
        shown_variants = ["plain", "last", "all", "ramp", "all_r2"]
        broken = []
        for s in range(len(worlds)):
            occ, grids, starts, goals = stack([worlds[s]], model.device)
            si, gi = grid.cells(starts), grid.cells(goals)
            h = torch.from_numpy(keep[(first, "plain", s)])[None].to(model.device)
            if float(bottleneck(h, ~grids, si, gi)) < ON_RIDGE:
                broken.append(s)
            if len(broken) == 12:
                break
        for p in range(0, len(broken), 4):
            fig, axes = plt.subplots(4, len(shown_variants), figsize=(11.69, 9.6))
            fig.suptitle(
                f"{first}: maps where plain breaks (blue = H>=0.5 on geodesic, red = gap)",
                fontsize=11,
                weight="bold",
            )
            for r, s in enumerate(broken[p : p + 4]):
                occ, grids, starts, goals = stack([worlds[s]], model.device)
                si, gi = grid.cells(starts), grid.cells(goals)
                for c, vn in enumerate(shown_variants):
                    h = torch.from_numpy(keep[(first, vn, s)])[None].to(model.device)
                    route = route_mask(ridge_cost(h), ~grids, si, gi)[0].cpu().numpy()
                    b = float(bottleneck(h, ~grids, si, gi))
                    b2 = float(hedging(h, ~grids, si, gi)[0])
                    draw_route(
                        axes[r, c],
                        worlds[s],
                        h[0].cpu().numpy(),
                        route,
                        b,
                        f"#{s} {vn}  b*₂={b2:.2f}",
                    )
            for ax in axes[len(broken[p : p + 4]) :].ravel():
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)

        # per-step x0_hat for two broken maps: plain vs all vs ramp_r2
        trace_variants = [v for v in VARIANTS if v.name in ("plain", "all", "ramp_r2")]
        for s in broken[:2]:
            world = worlds[s]
            occ, grids, starts, goals = stack([world], model.device)
            fig, axes = plt.subplots(3, 5, figsize=(11.69, 7.4))
            fig.suptitle(
                f"map #{s}: x̂0 per step (cyan = cells raised by projection)",
                fontsize=11,
                weight="bold",
            )
            seed = (s // args.batch) * args.batch
            for r, v in enumerate(trace_variants):
                h, _, frames = sample(model, occ, grids, starts, goals, seed, v, trace=True)
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
                    axes[r, c].set_title(f"{v.name} step {c + 1} x̂0", fontsize=8)
                base_axes(axes[r, 4], world, h[0].cpu().numpy())
                axes[r, 4].set_title(f"{v.name} output", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
