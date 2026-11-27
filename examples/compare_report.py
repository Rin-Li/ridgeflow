"""PDF: the same maps under several checkpoints - where the ridge breaks, and route diversity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt  # noqa: E402
from connectivity_eval import bottleneck  # noqa: E402
from make_dataset import RRTStar  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from topology_report import base_axes  # noqa: E402  (shared plotting helpers)

from ridgeflow import GridSpec, RidgeModel, sample_rect_world, stack  # noqa: E402
from ridgeflow.model.targets import path_heatmap, world_to_pixel  # noqa: E402
from ridgeflow.training.gap import ON_RIDGE, geodesic_mask  # noqa: E402


def draw(ax, world, h, route, b, title, compact=False):
    base_axes(ax, world, h)
    cell = world.grid.cell
    gaps = route & (h < ON_RIDGE)
    on = route & ~gaps
    for mask, colour, size in ((on, "#3fa9ff", 3), (gaps, "#ff3b3b", 9)):
        xs, ys = np.nonzero(mask)
        ax.scatter(
            (xs + 0.5) * cell, (ys + 0.5) * cell, s=size, c=colour, marker="s", lw=0, zorder=6
        )
    sep = "  " if compact else "\n"
    ax.set_title(f"{title}{sep}b*={b:.2f} gaps {int(gaps.sum())}", fontsize=7 if compact else 8)


def sample_all(models, worlds, seed):
    occ, grids, starts, goals = stack(worlds, "cuda")
    grid = worlds[0].grid
    si = torch.floor(starts / grid.cell).long()
    gi = torch.floor(goals / grid.cell).long()
    out = {}
    for name, model in models.items():
        h = model.sample(occ, starts, goals, seed=seed)
        out[name] = (
            h.cpu().numpy(),
            geodesic_mask(h, ~grids, si, gi).cpu().numpy(),
            bottleneck(h, ~grids, si, gi).cpu().numpy(),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="name=path pairs, first is the reference")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--broken", type=int, default=24, help="maps to show")
    parser.add_argument("--output", default="outputs/compare_report.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    models = {}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        models[name] = RidgeModel(path, grid)
    names = list(models)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]

    # maps where the reference model's benchmark-seed ridge is broken
    picked, counts = [], {n: 0 for n in names}
    for begin in range(0, len(worlds), 250):
        chunk = worlds[begin : begin + 250]
        res = sample_all(models, chunk, seed=begin)
        for n in names:
            counts[n] += int((res[n][2] < ON_RIDGE).sum())
        for k in np.flatnonzero(res[names[0]][2] < ON_RIDGE):
            picked.append((begin + int(k), {n: tuple(v[k] for v in res[n]) for n in names}))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = len(names)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(
            0.5, 0.94, "Ridge comparison before/after fine-tuning",
            ha="center", fontsize=17, weight="bold",
        )
        lines = [
            f"{len(worlds)} benchmark maps, one sample per model with the same seed.",
            "",
            "Broken samples (b* < 0.5, i.e. no route with H ≥ 0.5 throughout between start and "
            "goal, no corner cutting):",
            *[f"    {n}: {counts[n]} / {len(worlds)}" for n in names],
            "",
            f"Following pages: maps where {names[0]} is broken ({len(picked)} in total, first "
            f"{args.broken} shown),",
            "one map per row, one model per column, same noise seed per map.",
            "  background = sampled ridge H; blue = geodesic cells with H ≥ 0.5; red = gaps on "
            "the geodesic (H < 0.5)",
            "  dark gray = obstacles, light gray = 1px padding",
            "",
            "Last pages: diversity. Same map with 4 noise seeds; the first row is 4 runs of the "
            "RRT* demo.",
        ]
        for i, s in enumerate(lines):
            fig.text(0.07, 0.87 - 0.026 * i, s, fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)

        shown = picked[: args.broken]
        for p in range(0, len(shown), 4):
            rows = shown[p : p + 4]
            fig, axes = plt.subplots(4, cols, figsize=(8.27, 11.0))
            axes = np.atleast_2d(axes)
            fig.suptitle(
                f"{names[0]} broken maps {p + 1}-{p + len(rows)}", fontsize=12, weight="bold"
            )
            for r, (seed, data) in enumerate(rows):
                for c, n in enumerate(names):
                    h, route, b = data[n]
                    draw(axes[r, c], worlds[seed], h, route, b, f"map #{seed} · {n}")
            for ax in axes[len(rows) :].ravel():
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

        from multiprocessing import Pool

        from connectivity_eval import _rrt

        with Pool(20) as pool:
            spread = pool.map(_rrt, [(s, 4) for s in range(120)])
        for seed in np.argsort(np.nan_to_num(spread, nan=-1.0))[::-1][:3].tolist():
            world = worlds[seed]
            fig, axes = plt.subplots(cols + 1, 4, figsize=(8.27, 2.4 * (cols + 1) + 0.6))
            fig.suptitle(
                f"Diversity: map #{seed} (one of the maps with the most varied RRT* routes), 4 "
                f"noise seeds",
                fontsize=12,
                weight="bold",
            )
            for j in range(4):
                path = RRTStar(world, rng=1000 * seed + j).plan()
                h = path_heatmap(world_to_pixel(path, grid), grid.size)[0].T.numpy()
                base_axes(axes[0, j], world, h)
                axes[0, j].set_title(f"RRT* run {j + 1}", fontsize=8)
            for j in range(4):
                res = sample_all(models, [world], seed=10_000 * j)
                for c, n in enumerate(names):
                    h, route, b = (v[0] for v in res[n])
                    draw(axes[c + 1, j], world, h, route, b, f"{n} s{j + 1}", compact=True)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
    print(counts)
    print(out)


if __name__ == "__main__":
    main()
