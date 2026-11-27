"""TMP64: does the sampled heatmap connect start and goal, and what does a break cost FOCAL?

For each checkpoint and sampling mode it reports, over the test queries and several seeds:
  connected   share of heatmaps with b* >= 0.5 (a free 8-connected route, no corner cutting,
              with H >= 0.5 on every cell; raw grid, no inflation)
  exp. red.   mean per-query reduction of FOCAL (w = 2) expansions against A*
  cost        mean cost ratio to the optimum
and splits the expansion reduction by connected / broken heatmaps. Sampling modes:
  K1          one draw (the shipped 4-step sampler)
  best-of-M   M draws in one batch, keep the one with the largest b* (no search involved)
  r2          every sampling step run twice (self-recurrence)
Writes a results PDF.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.common import (
    BY_NAME,
    PLAIN,
    TMP64_DATA,
    astar_baseline,
    draw_search,
    search,
    tmp64_heatmaps,
    tmp64_queries,
)
from ridgeflow import GridSpec, RidgeModel
from ridgeflow.routes import ON_RIDGE
from ridgeflow.search import focal

RECUR = BY_NAME["plain_r2"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoints", nargs="+", help="name=path")
    parser.add_argument("--data", default=TMP64_DATA)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=0, help="0: all")
    parser.add_argument("--output", default="outputs/tmp64_report.pdf")
    parser.add_argument("--json", default=None, help="also write the table rows here")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True).item()
    blocked, sc, gc = tmp64_queries(data, args.queries)
    n = len(blocked)
    pool = Pool(20)
    a_exp, a_cost = astar_baseline(pool, blocked, sc, gc)
    print(f"A*: mean expansions {a_exp.mean():.0f}", flush=True)

    modes = [("K1", PLAIN, 1), ("best-of-4", PLAIN, 4), ("best-of-8", PLAIN, 8), ("r2", RECUR, 1)]
    rows, keep = [], {}
    grid = GridSpec()
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model = RidgeModel(path, grid)
        for mode, variant, draws in modes:
            red, cost, conn, red_ok, red_bad, gen, cross = [], [], [], [], [], 0.0, []
            for seed in range(args.seeds):
                for begin in range(0, n, args.batch):
                    idx = np.arange(begin, min(begin + args.batch, n))
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    h, b, walls = tmp64_heatmaps(
                        model, data, idx, 1000 * seed + begin, variant, draws
                    )
                    pick = b.argmax(1)
                    rows_ = torch.arange(len(idx), device=h.device)
                    hp, bp = h[rows_, pick], b[rows_, pick]
                    bw = walls[rows_, pick]
                    cross += ((bp < ON_RIDGE) & (bw >= ON_RIDGE)).tolist()
                    torch.cuda.synchronize()
                    gen += time.perf_counter() - t1
                    hn = hp.cpu().numpy()
                    res = pool.map(
                        search,
                        [(blocked[i], sc[i], gc[i], hn[k], args.w) for k, i in enumerate(idx)],
                    )
                    ok = (bp >= ON_RIDGE).cpu().numpy()
                    for k, i in enumerate(idx):
                        r = 100 * (1 - res[k][0] / a_exp[i])
                        red.append(r)
                        cost.append(res[k][1] / a_cost[i])
                        conn.append(ok[k])
                        (red_ok if ok[k] else red_bad).append(r)
                        if seed == 0 and mode == "K1":
                            keep[(name, int(i))] = (hn[k], bool(ok[k]), r)
            row = dict(
                model=name,
                mode=mode,
                conn=100 * np.mean(conn),
                cross=100 * np.mean(cross),
                red=np.mean(red),
                red_sd=np.std(red) / np.sqrt(len(red)),
                cost=np.mean(cost),
                red_ok=np.mean(red_ok) if red_ok else float("nan"),
                red_bad=np.mean(red_bad) if red_bad else float("nan"),
                gen_ms=1000 * gen / (n * args.seeds),
            )
            rows.append(row)
            print(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                flush=True,
            )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}
                    for r in rows
                ],
                ensure_ascii=False,
                indent=1,
            )
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "TMP64: heatmap connectivity and FOCAL expansion",
            ha="center",
            fontsize=16,
            weight="bold",
        )
        lines = [
            f"{n} test queries × {args.seeds} seeds; FOCAL w={args.w}, "
            "8-connected, no corner cutting, raw grid.",
            f"A* expands {a_exp.mean():.0f} nodes on average (Python, 20 processes).",
            "Connected = a route with H≥0.5 throughout exists between start and goal; expansion "
            "cut = mean ± s.e. of 1−FOCAL/A*; ",
            "Cost = path cost / optimal; wall-crossing = share of samples that are disconnected "
            "but connected if obstacles are ignored (ridge passes through a wall).",
            "best-of-M = draw M samples per batch and keep the largest b* (no search); r2 = run "
            "each sampling step twice.",
            "Generation time is amortized per query over the GPU batch.",
        ]
        for i, s in enumerate(lines):
            fig.text(0.04, 0.88 - 0.035 * i, s, fontsize=9.5)
        cells = [
            [
                r["model"],
                r["mode"],
                f"{r['conn']:.1f}%",
                f"{r['cross']:.1f}%",
                f"{r['red']:.1f} ± {r['red_sd']:.1f}",
                f"{r['cost']:.3f}",
                f"{r['red_ok']:.1f}",
                f"{r['red_bad']:.1f}",
                f"{r['gen_ms']:.2f}",
            ]
            for r in rows
        ]
        ax = fig.add_axes([0.04, 0.02, 0.92, 0.66])
        ax.axis("off")
        tab = ax.table(
            cellText=cells,
            colLabels=[
                "model",
                "sampler",
                "connected",
                "wall-cross",
                "expansion cut %",
                "cost ratio",
                "cut % (conn.)",
                "cut % (disc.)",
                "gen. ms",
            ],
            loc="upper center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.35)
        pdf.savefig(fig)
        plt.close(fig)

        first = args.checkpoints[0].split("=", 1)[0]
        broken = [i for (m, i), v in sorted(keep.items()) if m == first and not v[1]][:12]
        for p in range(0, len(broken), 6):
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.0))
            fig.suptitle(
                f"{first}: disconnected-heatmap queries (light blue = nodes expanded by FOCAL)",
                fontsize=11,
                weight="bold",
            )
            for ax, i in zip(axes.ravel(), broken[p : p + 6], strict=False):
                h, _, r = keep[(first, i)]
                res = focal(blocked[i], sc[i], gc[i], h, args.w, keep_closed=True)
                title = f"#{i}  expansion cut {r:.0f}%"
                draw_search(ax, blocked[i], sc[i], gc[i], h, res.closed, None, title)
            for ax in axes.ravel()[len(broken[p : p + 6]) :]:
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    pool.close()
    print(out)


if __name__ == "__main__":
    main()
