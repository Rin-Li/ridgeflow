"""Ridge walker on last-step-projected heatmaps: success, failure categories, timing, PDF.

    python examples/walker_eval.py                       # current walker
    python examples/walker_eval.py --compare old=...     # see --help

Failure categories (walker never collides: every step is collision-checked):
  overshoot  came within 3 px of the goal but never connected to it
  stuck      every candidate in the forward cone was blocked
  lost       used up max_steps (wandering once the ridge energy ran out)
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

sys.path.insert(0, str(Path(__file__).parent))
from sampler_ablation import VARIANTS, sample  # noqa: E402
from topology_report import base_axes  # noqa: E402  (registers the CJK font)

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402
from ridgeflow.planners.walker import WalkerConfig  # noqa: E402
from ridgeflow.routes import project  # noqa: E402

PLAIN = next(v for v in VARIANTS if v.name == "plain")


def classify(world, path, ok, max_steps):
    if ok:
        return "ok"
    near = np.linalg.norm(path - world.goal, axis=1).min() / world.grid.cell
    if near <= 3.0:
        return "overshoot"
    if len(path) >= max_steps + 1:
        return "lost"
    return "stuck"


def turning(path: np.ndarray) -> tuple[float, float]:
    """Total and largest heading change (rad) along a polyline."""
    d = np.diff(path, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    if len(d) < 2:
        return 0.0, 0.0
    a = np.arctan2(d[:, 1], d[:, 0])
    t = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(t.sum()), float(t.max())


class GeodesicReference:
    """Not a walker: the ridge geodesic's grid route, for comparing smoothness."""

    def __init__(self, grid):
        self.grid = grid
        self.config = WalkerConfig()

    def walk(self, h, grids, starts, goals, route=None):
        from ridgeflow.planners.common import Result
        from ridgeflow.routes import ridge_cost, route_paths

        cell = self.grid.cell
        si = torch.floor(starts / cell).long()
        gi = torch.floor(goals / cell).long()
        paths, ok = [], []
        for c, s_, g_ in zip(
            route_paths(ridge_cost(h), ~grids, si, gi),
            starts.cpu().numpy(),
            goals.cpu().numpy(),
            strict=True,
        ):
            if c is None:
                paths.append(s_[None])
                ok.append(False)
            else:
                paths.append(np.concatenate([s_[None], (c[1:-1] + 0.5) * cell, g_[None]]))
                ok.append(True)
        return Result(paths=paths, reached=np.array(ok))


def run(walkers: dict, model, worlds, k, batch, projected: bool):
    """Walkers whose name contains ``+route`` also get the projection's route."""
    grid = model.grid
    rec = {
        n: {"cat": [], "len": [], "secs": 0.0, "fixed": 0, "turn": [], "kink": [], "bt": 0}
        for n in walkers
    }
    keep = {}
    for begin in range(0, len(worlds), batch):
        chunk = worlds[begin : begin + batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si = torch.floor(starts / grid.cell).long()
        gi = torch.floor(goals / grid.cell).long()
        for j in range(k):
            h, _, _ = sample(model, occ, grids, starts, goals, begin + 10_000 * j, PLAIN)
            route = None
            if projected:  # = projecting the last step's x0_hat, which is the output
                h, route = project(h, ~grids, si, gi)
            for name, walker in walkers.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                use = route if "+route" in name else None
                res = walker.walk(h, grids, starts, goals, route=use)
                torch.cuda.synchronize()
                rec[name]["secs"] += time.perf_counter() - t0
                if getattr(res, "repaired", None) is not None:
                    rec[name]["fixed"] += int(np.sum(res.repaired))
                if getattr(res, "backtracks", None) is not None:
                    rec[name]["bt"] += int(np.sum(res.backtracks))
                for q, (w, p, r) in enumerate(zip(chunk, res.paths, res.reached, strict=True)):
                    ok = bool(r) and w.path_free(p)
                    cat = classify(w, p, ok, walker.config.max_steps)
                    rec[name]["cat"].append(cat)
                    if ok:
                        rec[name]["len"].append(
                            float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
                        )
                        total, kink = turning(p)
                        rec[name]["turn"].append(total)
                        rec[name]["kink"].append(kink)
                    fixed = getattr(res, "repaired", None)
                    shown = "repaired" if ok and fixed is not None and fixed[q] else cat
                    if j == 0:
                        keep.setdefault(begin + q, {"h": h[q].cpu().numpy()})[name] = (p, shown)
    return rec, keep


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/ridgeflow_rrt64.pt")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--output", default="outputs/walker_eval.pdf")
    parser.add_argument("--focus", default=None, help="whose failures to draw")
    parser.add_argument(
        "--geodesic", action="store_true", help="add the grid geodesic as reference"
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="extra walkers as name=field:value,field:value (WalkerConfig)",
    )
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid)
    walkers = {
        "original": RidgeWalker(
            grid,
            WalkerConfig(arrive_px=1.0, reverse_when_stuck=False, repair=False, backend="torch"),
        )
    }
    for item in args.configs:
        name, spec = item.split("=", 1)
        fields = {}
        for kv in filter(None, spec.split(",")):
            key, val = kv.split(":")
            default = getattr(WalkerConfig(), key)
            if isinstance(default, bool):
                fields[key] = val == "1"
            elif default is None:
                fields[key] = float(val)
            else:
                fields[key] = type(default)(val)
        walkers[name] = RidgeWalker(grid, WalkerConfig(**fields))
    if args.geodesic:
        walkers["geodesic (ref)"] = GeodesicReference(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    occ, grids, starts, goals = stack(worlds[:4], model.device)
    for w in walkers.values():  # warm-up
        w.walk(sample(model, occ, grids, starts, goals, 0, PLAIN)[0], grids, starts, goals)
    rec, keep = run(walkers, model, worlds, args.k, args.batch, not args.no_projection)

    n = len(worlds) * args.k
    rows = []
    for name, r in rec.items():
        cats = np.array(r["cat"])
        row = (
            [name, f"{100 * np.mean(cats == 'ok'):.2f}%"]
            + [str(int((cats == c).sum())) for c in ("overshoot", "stuck", "lost")]
            + [
                str(r["fixed"]),
                str(r["bt"]),
                f"{np.mean(r['len']):.2f}",
                f"{np.mean(r['turn']):.2f}",
                f"{np.degrees(np.percentile(r['kink'], 95)):.0f}°",
                f"{1000 * r['secs'] / n:.2f}",
            ]
        )
        rows.append(row)
        print(row, flush=True)

    # single-query latency: projection (when used) + walk
    occ, grids, starts, goals = stack(worlds[:1], model.device)
    si = torch.floor(starts / grid.cell).long()
    gi = torch.floor(goals / grid.cell).long()
    h1 = sample(model, occ, grids, starts, goals, 0, PLAIN)[0]
    for name, w in walkers.items():
        ts = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h, route = (h1, None) if args.no_projection else project(h1, ~grids, si, gi)
            w.walk(h, grids, starts, goals, route=route if "+route" in name else None)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        print(f"{name}: single-query projection + walk {1000 * np.median(ts):.1f} ms")
        next(r for r in rows if r[0] == name).append(f"{1000 * np.median(ts):.1f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = list(walkers)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "Walker on projected heatmaps", ha="center", fontsize=16, weight="bold")
        lines = [
            f"{len(worlds)} benchmark maps × {args.k} noise seeds; heatmap = base-model sample + "
            f"{'no projection' if args.no_projection else 'last-step L2 projection'}; "
            "success = reached the goal without touching padding.",
            "Failure types: overshoot = got within 3px of the goal but never connected; stuck = "
            "every candidate in the forward cone blocked; "
            "lost = step budget exhausted (wandering after energy ran out). Patched = failures "
            "repaired along the geodesic and straightened (counted as success).",
            "Total turn = sum of heading changes (rad); max turn p95 = 95th percentile of each "
            "path's largest single turn.",
        ]
        for i, s in enumerate(lines):
            fig.text(0.04, 0.88 - 0.035 * i, s, fontsize=10)
        ax = fig.add_axes([0.06, 0.3, 0.88, 0.45])
        ax.axis("off")
        tab = ax.table(
            cellText=rows,
            colLabels=[
                "walker",
                "success rate",
                "overshoot",
                "stuck",
                "lost",
                "patched",
                "backtracks",
                "path length",
                "total turn",
                "max turn p95",
                "ms/map (batch)",
                "ms (single)",
            ],
            loc="upper center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        tab.scale(1, 2.0)
        pdf.savefig(fig)
        plt.close(fig)

        # failures of the first walker, with every walker's path on the same heatmap
        focus = args.focus or names[0]
        fails = [s for s in sorted(keep) if keep[s][focus][1] != "ok"]  # failed or repaired
        colours = ["#00e68a", "#ffd400", "#3fa9ff", "#ff5c5c"]
        for p in range(0, min(len(fails), 24), 6):
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.0))
            fig.suptitle(
                f"{focus} failed/patched maps ({p + 1}-{min(p + 6, len(fails))} / {len(fails)})  "
                + "  ".join(f"{n}={c}" for n, c in zip(names, colours, strict=False)),
                fontsize=10,
                weight="bold",
            )
            for ax, s in zip(axes.ravel(), fails[p : p + 6], strict=False):
                base_axes(ax, worlds[s], keep[s]["h"])
                title = []
                for name, c in zip(names, colours, strict=False):
                    path, cat = keep[s][name]
                    ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.6, zorder=5)
                    if cat not in ("ok", "repaired"):
                        ax.plot(*path[-1], "x", color=c, ms=10, mew=2.5, zorder=6)
                    if cat == "repaired":
                        ax.plot(path[:, 0], path[:, 1], "o", color=c, ms=3, zorder=6)
                    title.append(f"{name}:{cat}")
                ax.set_title(f"#{s}  " + "  ".join(title), fontsize=8)
            for ax in axes.ravel()[len(fails[p : p + 6]) :]:
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
