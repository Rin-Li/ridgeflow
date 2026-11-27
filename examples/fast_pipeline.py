"""K candidate routes for one query, default vs fast pipeline (ridgeflow.fast).

Pipeline: K draws in one batch -> L2 projection of each -> backtracking walker on each.
Default = RidgeModel (fp32, eager) + RidgeWalker; fast = FastRidgeModel (bf16, torch.compile)
+ FastRidgeWalker. Reports the per-stage latency of single queries for several K, and the
quality of both over the benchmark worlds (projection keeps every draw connected; success =
reached and collision-free). Writes a PDF.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402
from ridgeflow.fast import FastRidgeModel, FastRidgeWalker  # noqa: E402
from ridgeflow.routes import ON_RIDGE, bottleneck, project  # noqa: E402


def plan(model, walker, occ, grids, starts, goals, seed):
    """Returns paths, reached, raw connectivity and the stage times (ms)."""
    cell = model.grid.cell
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = model.sample(occ, starts, goals, seed)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    si = torch.floor(starts / cell).long()
    gi = torch.floor(goals / cell).long()
    hp, route = project(h, ~grids, si, gi)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    res = walker.walk(hp, grids, starts, goals, route=route)
    t3 = time.perf_counter()
    conn = (bottleneck(h, ~grids, si, gi) >= ON_RIDGE).cpu().numpy()
    return res, conn, (1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t3 - t2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/runs/rrtc_full/best.pt")
    parser.add_argument("--ks", default="1,4,8,16")
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=4, help="draws per world in the quality run")
    parser.add_argument("--output", default="outputs/fast_pipeline.pdf")
    args = parser.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    grid = GridSpec()
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    pipelines = {
        "default (fp32 + old walker)": (RidgeModel(args.model, grid), RidgeWalker(grid)),
        "fast (bf16 + compile + new walker)": (
            FastRidgeModel(args.model, grid),
            FastRidgeWalker(grid),
        ),
    }
    t0 = time.perf_counter()
    pipelines["fast (bf16 + compile + new walker)"][0].warmup(sorted({*ks, args.k * 50}))
    print(f"compile warm-up {time.perf_counter() - t0:.0f} s", flush=True)

    latency = {}
    for name, (model, walker) in pipelines.items():
        for k in ks:
            stages = []
            for q in range(args.latency_queries + 3):
                occ, grids, starts, goals = (
                    t.repeat_interleave(k, 0) for t in stack([worlds[q]], model.device)
                )
                _, _, st = plan(model, walker, occ, grids, starts, goals, q)
                if q >= 3:
                    stages.append(st)
            latency[(name, k)] = np.median(np.array(stages), axis=0)
            print(name, k, np.round(latency[(name, k)], 2), flush=True)

    quality = {}
    for name, (model, walker) in pipelines.items():
        ok, conn, lengths = [], [], []
        for begin in range(0, len(worlds), 50):
            chunk = worlds[begin : begin + 50]
            occ, grids, starts, goals = (
                t.repeat_interleave(args.k, 0) for t in stack(chunk, model.device)
            )
            res, c, _ = plan(model, walker, occ, grids, starts, goals, begin)
            conn += c.tolist()
            for i, (p, r) in enumerate(zip(res.paths, res.reached, strict=True)):
                good = bool(r) and chunk[i // args.k].path_free(p)
                ok.append(good)
                if good:
                    lengths.append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
        quality[name] = (100 * np.mean(conn), 100 * np.mean(ok), float(np.mean(lengths)))
        print(name, np.round(quality[name], 3), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = list(pipelines)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("K routes per query: default vs fast (RTX 5080)", fontsize=14, weight="bold")
        ax = fig.add_axes([0.07, 0.42, 0.88, 0.45])
        width = 0.38
        colours = ("#4c78a8", "#f58518", "#54a24b")
        stage_names = ("generation", "projection", "walker")
        for j, name in enumerate(names):
            bottom = np.zeros(len(ks))
            for s in range(3):
                v = np.array([latency[(name, k)][s] for k in ks])
                ax.bar(np.arange(len(ks)) + (j - 0.5) * width, v, width, bottom=bottom,
                       color=colours[s], alpha=1.0 if j else 0.45,
                       label=f"{stage_names[s]}" if j else None)  # fmt: skip
                bottom += v
            for i, tot in enumerate(bottom):
                ax.text(i + (j - 0.5) * width, tot + 0.4, f"{tot:.1f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(ks)), [f"K={k}\n(left default / right fast)" for k in ks])
        ax.set_ylabel("median latency (ms)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        cells = []
        for k in ks:
            a, b = latency[(names[0], k)], latency[(names[1], k)]
            cells.append([f"K={k}", *(f"{x:.2f} → {y:.2f}" for x, y in zip(a, b, strict=True)),
                          f"{a.sum():.2f} → {b.sum():.2f}({a.sum() / b.sum():.1f}x)"])  # fmt: skip
        for name in names:
            c, o, ln = quality[name]
            cells.append([name, f"conn. {c:.2f}%", f"success {o:.2f}%", f"length {ln:.3f} m", ""])
        tab = fig.add_axes([0.05, 0.02, 0.9, 0.3])
        tab.axis("off")
        t = tab.table(cellText=cells, colLabels=["", "gen. ms", "proj. ms", "walker ms", "sum ms"],
                      loc="upper center", cellLoc="center")  # fmt: skip
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.4)
        fig.text(0.05, 0.005, f"Latency = median over {args.latency_queries} queries; "
                 f"quality = {args.queries} maps × {args.k} samples (same seed on both sides; "
                 "bf16 sends a few samples down another route).", fontsize=8)  # fmt: skip
        pdf.savefig(fig)
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
