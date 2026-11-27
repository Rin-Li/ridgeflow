"""Does the flow model matter once the last step is projected? Compare, on the same maps:

  empty       H = 0 everywhere, then the same L2 projection (= plain shortest free path)
  model       the shipped sampler, last-step L2 projection
  model_plain the shipped sampler, no projection
  RRT*        the demonstrations' planner (what the model was trained to imitate)

and decode each heatmap with the ridge geodesic. Success saturates by construction, so the
report looks at what success hides: how much of the route the projection had to invent, path
length against the shortest path, clearance, smoothness, and route diversity across seeds.
Writes ``outputs/value_of_model.pdf``.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import distance_transform_edt

from experiments.common import (
    RRT_CHECKPOINT,
    base_axes,
    diversity,
    geodesic_paths,
    path_length,
    turning,
)
from ridgeflow import GridSpec, RidgeModel, sample_rect_world, stack
from ridgeflow.model.targets import path_heatmap, world_to_pixel
from ridgeflow.routes import (
    ON_RIDGE,
    project,
    ridge_cost,
    route_mask,
)


def path_stats(world, pts, clearance):
    """Length (world units), mean / min clearance (cells), total turning (rad)."""
    length = path_length(pts)
    idx = np.clip(np.floor(pts / world.grid.cell).astype(int), 0, world.grid.size - 1)
    clr = clearance[idx[:, 0], idx[:, 1]]
    # smooth a little before measuring turning so grid zig-zags do not dominate
    if len(pts) > 4:
        k = np.ones(3) / 3
        sm = np.stack([np.convolve(pts[:, i], k, mode="valid") for i in range(2)], -1)
    else:
        sm = pts
    return length, float(clr.mean()), float(clr.min()), turning(sm)[0]


def _rrt(args):
    from ridgeflow.planners import RRTStar

    torch.set_num_threads(1)
    seed, j = args
    world = sample_rect_world(np.random.default_rng(seed), GridSpec())
    return RRTStar(world, rng=1000 * seed + j).plan()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", default=RRT_CHECKPOINT)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output", default="outputs/value_of_model.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    clear = [distance_transform_edt(~w.occupancy().numpy()) for w in worlds]
    methods = ("empty", "model", "model_plain", "RRT*")
    stats = {
        m: {"ok": [], "len": [], "clr": [], "clrmin": [], "turn": [], "div": []} for m in methods
    }
    invented = []  # share of the model's decoded route cells that the projection lifted
    keep = {}

    with Pool(20) as pool:
        rrt = pool.map(_rrt, [(s, j) for j in range(args.k) for s in range(args.queries)])
    rrt = [rrt[j * args.queries : (j + 1) * args.queries] for j in range(args.k)]

    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si, gi = grid.cells(starts), grid.cells(goals)
        masks = {m: [] for m in methods}
        for j in range(args.k):
            seed = begin + 10_000 * j
            h_plain = model.sample(occ, starts, goals, seed)
            h_model, route = project(h_plain, ~grids, si, gi)
            lifted = route & (h_plain < ON_RIDGE)
            # identical to the "last" variant: projection of the final x0_hat
            h_empty, _ = project(torch.zeros_like(h_plain), ~grids, si, gi)
            heat = {"empty": h_empty, "model": h_model, "model_plain": h_plain}
            paths = {m: geodesic_paths(h, grids, starts, goals, grid) for m, h in heat.items()}
            paths["RRT*"] = rrt[j][begin : begin + len(chunk)]
            for m in methods:
                for q, (w, p) in enumerate(zip(chunk, paths[m], strict=True)):
                    ok = p is not None and w.path_free(p)
                    stats[m]["ok"].append(ok)
                    if ok:
                        L, c, cmin, turn = path_stats(w, p, clear[begin + q])
                        stats[m]["len"].append(L)
                        stats[m]["clr"].append(c)
                        stats[m]["clrmin"].append(cmin)
                        stats[m]["turn"].append(turn)
                    if j == 0:
                        keep.setdefault(begin + q, {})[m] = p
                if m == "RRT*":
                    masks[m].append(
                        np.stack(
                            [
                                (
                                    path_heatmap(world_to_pixel(p, grid), grid.size)[0].T.numpy()
                                    >= ON_RIDGE
                                )
                                if p is not None
                                else np.zeros((grid.size, grid.size), bool)
                                for p in paths[m]
                            ]
                        )
                    )
                else:
                    masks[m].append((heat[m] >= ON_RIDGE).cpu().numpy())
            route = route_mask(ridge_cost(h_model), ~grids, si, gi)
            n_route = route.sum((1, 2)).clamp_min(1)
            invented += ((lifted & route).sum((1, 2)) / n_route).cpu().tolist()
            if j == 0:
                hm, he = h_model.cpu().numpy(), h_empty.cpu().numpy()
                for q in range(len(chunk)):
                    keep[begin + q]["h_model"] = hm[q]
                    keep[begin + q]["h_empty"] = he[q]
        for m in methods:
            stats[m]["div"] += [diversity(x) for x in np.stack(masks[m], 1)]
        print(f"{begin + len(chunk)}/{len(worlds)}", flush=True)

    # length relative to the shortest free path (the empty-heatmap route) on the same map/seed
    rows = []
    for m in methods:
        s = stats[m]
        rows.append(
            [
                m,
                f"{100 * np.mean(s['ok']):.1f}%",
                f"{np.mean(s['len']):.2f}",
                f"{np.mean(s['clr']):.2f}",
                f"{np.mean(s['clrmin']):.2f}",
                f"{np.mean(s['turn']):.2f}",
                f"{np.mean(s['div']):.3f}",
            ]
        )
        print(rows[-1], flush=True)
    inv = np.array(invented)
    print(
        f"model: share of route cells invented by projection: mean {100 * inv.mean():.2f}%, "
        f"maps with any {100 * np.mean(inv > 0):.1f}%, median among those "
        f"{100 * np.median(inv[inv > 0]) if (inv > 0).any() else 0:.1f}%"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(
            0.5,
            0.95,
            "What does flow matching still contribute after projection?",
            ha="center",
            fontsize=16,
            weight="bold",
        )
        lines = [
            f"{len(worlds)} benchmark maps × {args.k} noise seeds; all decoded with the geodesic; "
            "success = reached the goal without touching an obstacle.",
            "empty = all-zero heatmap + last-step L2 projection (equivalent to the free-space "
            "shortest path); model = sample + last-step L2 projection; "
            "model_plain = no projection; RRT* = planner used for the training demos.",
            "Wall clearance: mean/min distance (cells) from path points to the raw obstacles; "
            "turn: total smoothed turning (rad); "
            "diversity: 1−IoU of routes across seeds on the same map.",
            f"Cells in model routes filled in by projection: mean {100 * inv.mean():.2f}%; "
            f"maps needing fill {100 * np.mean(inv > 0):.1f}%; "
            f"of those, median {100 * np.median(inv[inv > 0]) if (inv > 0).any() else 0:.1f}%.",
        ]
        for i, s in enumerate(lines):
            fig.text(0.04, 0.88 - 0.035 * i, s, fontsize=10)
        ax = fig.add_axes([0.08, 0.2, 0.84, 0.45])
        ax.axis("off")
        tab = ax.table(
            cellText=rows,
            colLabels=[
                "method",
                "success rate",
                "path length",
                "mean clearance",
                "min clearance",
                "total turn",
                "diversity",
            ],
            loc="upper center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(11)
        tab.scale(1, 2.0)
        pdf.savefig(fig)
        plt.close(fig)

        # side-by-side routes on the first maps
        colours = {"empty": "#ff5c5c", "model": "#00e68a", "RRT*": "#3fa9ff"}
        for p in range(0, 12, 6):
            fig, axes = plt.subplots(3, 4, figsize=(11.69, 8.6))
            fig.suptitle(
                "red = empty-heatmap projection (shortest path)  green = model + projection  blue "
                "= RRT*  "
                "(left: model heatmap, right: empty heatmap after projection)",
                fontsize=11,
                weight="bold",
            )
            for r, s in enumerate(range(p, p + 6)):
                ax_m, ax_e = axes[r // 2, (r % 2) * 2], axes[r // 2, (r % 2) * 2 + 1]
                base_axes(ax_m, worlds[s], keep[s]["h_model"])
                base_axes(ax_e, worlds[s], keep[s]["h_empty"])
                for ax in (ax_m, ax_e):
                    for m, c in colours.items():
                        if keep[s][m] is not None:
                            ax.plot(
                                keep[s][m][:, 0], keep[s][m][:, 1], "-", color=c, lw=1.5, zorder=5
                            )
                ax_m.set_title(f"map #{s} model heatmap", fontsize=8)
                ax_e.set_title(f"map #{s} empty-heatmap projection", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
