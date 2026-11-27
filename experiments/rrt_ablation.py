"""RRT (continuous) ablation: does each heatmap connect, and what does the walker make of it?

Over the benchmark worlds (``sample_rect_world`` seeds 0..N-1) and K draws per world:
  connected    b* >= 0.5 on the padded planning grid (8-connected, no corner cutting)
  walls        broken, but connected once obstacles are ignored (the ridge crosses a wall)
  gap cells    cells on the ridge geodesic with H < 0.5
  hedging      a second unbroken route outside a 3 px corridor around the main one
  diversity    mean pairwise 1 - IoU of the K ridge masks of one world
Walker success (reached and collision-free) for the two walkers on raw and on projected heatmaps:
  plain        the original walker (no reversing, no backtracking)
  dfs          the backtracking walker; on projected heatmaps it also gets the projection's route
plus length and total turning of the dfs walker's paths on projected heatmaps, and timings.
Writes a PDF (tables and example heatmaps) and, with --json, the rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))
from connectivity_eval import diversity  # noqa: E402
from sampler_ablation import VARIANTS, sample  # noqa: E402
from topology_report import base_axes  # noqa: E402  (registers the CJK font)
from walker_eval import turning  # noqa: E402

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402
from ridgeflow.planners.walker import WalkerConfig  # noqa: E402
from ridgeflow.routes import ON_RIDGE, bottleneck, hedging, project  # noqa: E402
from ridgeflow.training.gap import geodesic_mask  # noqa: E402

PLAIN = next(v for v in VARIANTS if v.name == "plain")
SHOW = 6  # example worlds drawn per checkpoint


def evaluate(model, worlds, k, batch, walkers):
    grid = model.grid
    r = {key: [] for key in ("conn", "walls", "gaps", "hedge", "div", "len", "turn")}
    for key in walkers:
        r[f"raw_{key}"], r[f"proj_{key}"] = [], []
    r["gen_s"] = r["proj_s"] = r["walk_s"] = 0.0
    shown = []
    for begin in range(0, len(worlds), batch):
        chunk = worlds[begin : begin + batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si = torch.floor(starts / grid.cell).long()
        gi = torch.floor(goals / grid.cell).long()
        masks = []
        for j in range(k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h, _, _ = sample(model, occ, grids, starts, goals, begin + 10_000 * j, PLAIN)
            torch.cuda.synchronize()
            r["gen_s"] += time.perf_counter() - t0
            b = bottleneck(h, ~grids, si, gi)
            bw = bottleneck(h, torch.ones_like(grids), si, gi)
            r["conn"] += (b >= ON_RIDGE).tolist()
            r["walls"] += ((b < ON_RIDGE) & (bw >= ON_RIDGE)).tolist()
            r["hedge"] += (hedging(h, ~grids, si, gi)[0] >= ON_RIDGE).tolist()
            r["gaps"] += (geodesic_mask(h, ~grids, si, gi) & (h < ON_RIDGE)).sum((1, 2)).tolist()
            masks.append((h >= ON_RIDGE).cpu().numpy())
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hp, route = project(h, ~grids, si, gi)
            torch.cuda.synchronize()
            r["proj_s"] += time.perf_counter() - t0
            for key, walker in walkers.items():
                for tag, hh, rt in (("raw", h, None), ("proj", hp, route)):
                    t0 = time.perf_counter()
                    res = walker.walk(hh, grids, starts, goals, route=rt if key == "dfs" else None)
                    if tag == "proj" and key == "dfs":
                        r["walk_s"] += time.perf_counter() - t0
                    for w, p, ok in zip(chunk, res.paths, res.reached, strict=True):
                        ok = bool(ok) and w.path_free(p)
                        r[f"{tag}_{key}"].append(ok)
                        if ok and tag == "proj" and key == "dfs":
                            r["len"].append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
                            r["turn"].append(turning(p)[0])
                    if j == 0 and begin == 0 and tag == "proj" and key == "dfs":
                        for q in range(SHOW):
                            shown.append((h[q].cpu().numpy(), res.paths[q], bool(b[q] >= ON_RIDGE)))
        r["div"] += [diversity(m) for m in np.stack(masks, 1)]
    n = len(worlds) * k
    row = {
        "connected": 100 * np.mean(r["conn"]),
        "walls": 100 * np.mean(r["walls"]),
        "gap_cells": float(np.mean(r["gaps"])),
        "hedging": 100 * np.mean(r["hedge"]),
        "diversity": float(np.mean(r["div"])),
        "length": float(np.mean(r["len"])),
        "turning": float(np.mean(r["turn"])),
        "gen_ms": 1000 * r["gen_s"] / n,
        "proj_ms": 1000 * r["proj_s"] / n,
        "walk_ms": 1000 * r["walk_s"] / n,
    }
    for key in walkers:
        row[f"raw_{key}"] = 100 * np.mean(r[f"raw_{key}"])
        row[f"proj_{key}"] = 100 * np.mean(r[f"proj_{key}"])
    return {k_: float(v) for k_, v in row.items()}, shown


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoints", nargs="+", help="name=path")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--json", default=None)
    parser.add_argument("--output", default="outputs/ablation/rrt.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    walkers = {
        "plain": RidgeWalker(
            grid, WalkerConfig(arrive_px=1.0, reverse_when_stuck=False, backtrack=False)
        ),
        "dfs": RidgeWalker(grid),
    }
    rows, shows = [], {}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model = RidgeModel(path, grid)
        if not rows:  # warm up numba and cuda once
            evaluate(model, worlds[:8], 1, 8, walkers)
        row, shown = evaluate(model, worlds, args.k, args.batch, walkers)
        rows.append({"model": name, **row})
        shows[name] = shown
        print(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()},
            flush=True,
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "RRT (continuous) ablation: heatmap connectivity and walker",
            ha="center",
            fontsize=16,
            weight="bold",
        )
        lines = [
            f"{args.queries} benchmark maps × {args.k} samples (4-step Euler); "
            "connected = an 8-connected, no-corner-cutting route with H>=0.5 throughout"
            " on the planning grid (conservative, no padding).",
            "through-wall = broken, but connected if obstacles are ignored; "
            "gap cells = geodesic cells with H<0.5; "
            "two routes = a second full route outside the 3px corridor; "
            "diversity = 1−IoU across samples of one map.",
            "walker success = reaches goal without collision. "
            "orig = no reversing/backtracking; "
            "backtrack = DFS walker (after projection, also gets a corridor "
            "around the projected route); "
            "projection = final-step L2 projection, guarantees connectivity.",
            "Path length / total turn: successful backtrack-walker paths "
            "on the projected heatmap (m / rad).",
        ]
        for i, s in enumerate(lines):
            fig.text(0.03, 0.88 - 0.035 * i, s, fontsize=8)
        topo = [
            [
                r["model"],
                f"{r['connected']:.1f}%",
                f"{r['walls']:.1f}%",
                f"{r['gap_cells']:.2f}",
                f"{r['hedging']:.1f}%",
                f"{r['diversity']:.3f}",
                f"{r['gen_ms']:.2f}",
            ]
            for r in rows
        ]
        walk = [
            [
                r["model"],
                f"{r['raw_plain']:.1f}%",
                f"{r['raw_dfs']:.1f}%",
                f"{r['proj_plain']:.1f}%",
                f"{r['proj_dfs']:.2f}%",
                f"{r['length']:.2f}",
                f"{r['turning']:.2f}",
                f"{r['proj_ms']:.2f}",
                f"{r['walk_ms']:.2f}",
            ]
            for r in rows
        ]
        for cells, labels, box in (
            (
                topo,
                [
                    "Model",
                    "Connected (single)",
                    "Through wall",
                    "Gap cells",
                    "Two routes",
                    "Diversity",
                    "Gen ms",
                ],
                [0.04, 0.38, 0.92, 0.3],
            ),
            (
                walk,
                [
                    "Model",
                    "Raw·orig",
                    "Raw·backtrack",
                    "Proj·orig",
                    "Proj·backtrack",
                    "Path length",
                    "Total turn",
                    "Proj ms",
                    "walker ms",
                ],
                [0.04, 0.02, 0.92, 0.33],
            ),
        ):
            ax = fig.add_axes(box)
            ax.axis("off")
            tab = ax.table(cellText=cells, colLabels=labels, loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(9)
            tab.scale(1, 1.45)
        pdf.savefig(fig)
        plt.close(fig)

        names = list(shows)
        fig, axes = plt.subplots(len(names), SHOW, figsize=(11.69, 2.0 * len(names) + 0.6))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            "Single-sample heatmaps, same maps "
            "(green = backtrack walker path after projection; ✓/✗ = raw heatmap connected)",
            fontsize=10,
        )
        for row_ax, name in zip(axes, names, strict=True):
            for q, ax in enumerate(row_ax):
                h, p, ok = shows[name][q]
                base_axes(ax, worlds[q], h)
                ax.plot(p[:, 0], p[:, 1], "-", color="#00e68a", lw=1.4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{name} #{q} {'✓' if ok else '✗'}", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
