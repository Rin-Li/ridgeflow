"""Case study on TMP64: one model (single draw, K1) against a reference model, query by query.

Pages: a summary (distributions of per-query expansion reduction and cost ratio), then queries
the model fixed (reference broken, model connected), queries still broken, and random queries.
Each panel shows the heatmap, the cells FOCAL expanded and the returned path.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.common import (
    PLAIN,
    TMP64_CHECKPOINT,
    TMP64_DATA,
    draw_search,
    tmp64_heatmaps,
    tmp64_queries,
)
from ridgeflow import GridSpec, RidgeModel
from ridgeflow.routes import ON_RIDGE
from ridgeflow.search import astar, focal


def _run(args):
    blocked, s, g, h, w = args
    a = astar(blocked, s, g)
    f = focal(blocked, s, g, h, w, keep_closed=True)
    return a.expanded, a.cost, f.expanded, f.cost, f.closed, f.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=f"full={TMP64_CHECKPOINT}")
    parser.add_argument("--reference", default="σ1.5=outputs/runs/tmp_s15/best.pt")
    parser.add_argument("--data", default=TMP64_DATA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--output", default="outputs/tmp64_cases.pdf")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True).item()
    blocked, sc, gc = tmp64_queries(data)
    n = len(blocked)
    grid = GridSpec()
    out = {}
    pool = Pool(20)
    for item in (args.reference, args.model):
        name, path = item.split("=", 1)
        model = RidgeModel(path, grid)
        hs, bs = [], []
        for begin in range(0, n, 128):
            idx = np.arange(begin, min(begin + 128, n))
            h, b, _ = tmp64_heatmaps(model, data, idx, 1000 * args.seed + begin, PLAIN, 1)
            hs.append(h[:, 0].cpu().numpy())
            bs.append(b[:, 0].cpu().numpy())
        h, b = np.concatenate(hs), np.concatenate(bs)
        res = pool.map(_run, [(blocked[i], sc[i], gc[i], h[i], args.w) for i in range(n)])
        red = np.array([100 * (1 - r[2] / r[0]) for r in res])
        cost = np.array([r[3] / r[1] for r in res])
        out[name] = dict(h=h, ok=b >= ON_RIDGE, b=b, res=res, red=red, cost=cost)
        print(
            f"{name}: connected {100 * out[name]['ok'].mean():.1f}%  "
            f"reduction {red.mean():.1f}%  cost {cost.mean():.3f}",
            flush=True,
        )
    pool.close()
    ref, mod = (item.split("=", 1)[0] for item in (args.reference, args.model))
    R, M = out[ref], out[mod]

    fixed = np.flatnonzero(~R["ok"] & M["ok"])
    still = np.flatnonzero(~M["ok"])
    worse = np.flatnonzero(R["ok"] & ~M["ok"])
    rng = np.random.default_rng(0)
    typical = rng.choice(np.flatnonzero(M["ok"]), 6, replace=False)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"TMP64 cases: {mod} (single sample) vs {ref}", fontsize=15, weight="bold")
        lines = [
            f"{n} test queries, seed {args.seed}, FOCAL w={args.w}. "
            "Orange = heatmap, light blue = FOCAL expansion, green = path, gray = obstacles.",
            f"{ref}: connected {100 * R['ok'].mean():.1f}%, exp. reduction {R['red'].mean():.1f}%, "
            f"cost ratio {R['cost'].mean():.3f}.",
            f"{mod}: connected {100 * M['ok'].mean():.1f}%, exp. reduction {M['red'].mean():.1f}%, "
            f"cost ratio {M['cost'].mean():.3f}.",
            f"Fixed ({ref} broken -> {mod} connected): {len(fixed)}; still broken: {len(still)}; "
            f"newly broken: {len(worse)}.",
            f"Fixed queries: exp. reduction {R['red'][fixed].mean():.1f}% → "
            f"{M['red'][fixed].mean():.1f}%.",
        ]
        for i, s in enumerate(lines):
            fig.text(0.05, 0.88 - 0.04 * i, s, fontsize=10)
        ax1 = fig.add_axes([0.07, 0.08, 0.4, 0.5])
        bins = np.linspace(-150, 100, 51)
        ax1.hist(R["red"], bins, alpha=0.6, label=ref, color="#999999")
        ax1.hist(M["red"], bins, alpha=0.6, label=mod, color="#ff8c1a")
        ax1.set_xlabel("Expansion reduction per query %")
        ax1.set_ylabel("Queries")
        ax1.legend()
        ax2 = fig.add_axes([0.56, 0.08, 0.4, 0.5])
        ax2.scatter(R["red"], M["red"], s=5, c=np.where(M["ok"], "#ff8c1a", "#3366ff"))
        ax2.plot([-150, 100], [-150, 100], "k--", lw=0.8)
        ax2.set_xlabel(f"{ref} expansion reduction %")
        ax2.set_ylabel(f"{mod} expansion reduction % (blue = still broken)")
        pdf.savefig(fig)
        plt.close(fig)

        def pages(ids, label):
            for p in range(0, min(len(ids), 12), 3):
                fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.4))
                fig.suptitle(
                    f"{label} ({p + 1}-{min(p + 3, len(ids), 12)} / {len(ids)})",
                    fontsize=11,
                    weight="bold",
                )
                for r, i in enumerate(ids[p : p + 3]):
                    for c, (nm, o) in enumerate(((ref, R), (mod, M))):
                        draw_search(
                            axes[r, c],
                            blocked[i],
                            sc[i],
                            gc[i],
                            o["h"][i],
                            *o["res"][i][4:],
                            f"#{i} {nm}  b*={o['b'][i]:.2f}  exp. red. {o['red'][i]:.0f}%  "
                            f"cost {o['cost'][i]:.3f}",
                        )
                for ax in axes[len(ids[p : p + 3]) :].ravel():
                    ax.axis("off")
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig)
                plt.close(fig)

        pages(
            fixed[np.argsort(M["red"][fixed] - R["red"][fixed])[::-1]],
            f"Queries broken under {ref}, connected under {mod}",
        )
        pages(still, f"Queries still broken under {mod}")
        pages(typical, f"Randomly drawn connected queries under {mod}")
    print(path)


if __name__ == "__main__":
    main()
