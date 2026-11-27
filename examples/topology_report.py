"""PDF report: each walker failure, the break in its ridge, and the stitched geodesic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from topology import bottleneck, geodesic  # noqa: E402

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
WALK, GEO, BRIDGE = "#00e68a", "#3fa9ff", "#ff3b3b"
ON_RIDGE = 0.5


def reach(mask: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """8-connected flood fill of ``mask`` (B,S,S) from one pixel per map."""
    rows = torch.arange(mask.shape[0], device=mask.device)
    x = torch.zeros_like(mask)
    x[rows, seed[:, 0], seed[:, 1]] = True
    x &= mask
    while True:
        grown = (F.max_pool2d(x.float()[:, None], 3, 1, 1)[:, 0] > 0.5) & mask
        if torch.equal(grown, x):
            return x
        x = grown


def base_axes(ax, world, image, cmap="magma", vmax=1.0):
    e = world.grid.extent
    ax.imshow(image.T, cmap=cmap, origin="lower", extent=[0, e, 0, e], vmin=0, vmax=vmax)
    occ = world.occupancy().numpy().T
    ax.imshow(
        np.ma.masked_where(~occ, occ),
        cmap="Greys",
        origin="lower",
        extent=[0, e, 0, e],
        vmin=0,
        vmax=1.4,
        alpha=0.95,
    )
    pad = (world.planning_grid() & ~world.occupancy()).numpy().T
    ax.imshow(
        np.ma.masked_where(~pad, pad),
        cmap=ListedColormap(["#777777"]),
        origin="lower",
        extent=[0, e, 0, e],
        alpha=0.35,
    )
    ax.plot(*world.start, "o", color="white", ms=7, mec="k", zorder=7)
    ax.plot(*world.goal, "*", color="cyan", ms=13, mec="k", zorder=7)
    ax.set_xlim(0, e)
    ax.set_ylim(0, e)
    ax.set_xticks([])
    ax.set_yticks([])


def geodesic_world(path_px, world, cell):
    return np.concatenate([world.start[None], (path_px[1:-1] + 0.5) * cell, world.goal[None]])


def summary_page(pdf, stats, n):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.5,
        0.95,
        "RidgeFlow topology repair: -log H geodesic stitching",
        ha="center",
        fontsize=17,
        weight="bold",
    )
    lines = [
        f"{n} random maps (benchmark.py seeds, 7-10 rectangles, line blocked), 4-step sample.",
        "",
        "Cost: c(v) = 1 − log max(H(v), τ), τ = 1e-3 (the walker's existing clamp).",
        "  Gaussian ridge H = exp(−d²/2σ²) => −log H = d²/2σ²: squared centerline dist. (σ units).",
        "  8-connected shortest path on the padded free grid (no corner cutting).",
        "",
        "Bottleneck b* = max_path min_v H(v) (weakest link start to goal): is the ridge broken?",
    ]
    for i, s in enumerate(lines):
        fig.text(0.07, 0.89 - 0.025 * i, s, fontsize=9)

    rows = [[k, f"{v:.1f}%"] for k, v in stats["rates"].items()]
    ax = fig.add_axes([0.12, 0.47, 0.76, 0.22])
    ax.axis("off")
    t = ax.table(
        cellText=rows,
        colLabels=["Method", "Success rate"],
        loc="center",
        cellLoc="left",
        colWidths=[0.75, 0.25],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.6)

    f = stats["fail"]
    diag = [
        "Walker failure diagnosis",
        f"  Failures: {f['n']}",
        f"  Ridge broken (b*<0.05): {f['broken']} ({100 * f['broken'] / f['n']:.0f}%)",
        f"    - cut only by 1px padding (connected on raw obstacles, wall-hugging): {f['wallhug']}",
        f"  Ridge connected but walker still fails: {f['n'] - f['broken']} (see last pages:",
        "    diagonal slits, where b* allows corner cutting",
        "    but the walker's collision check does not; or goal on map edge, walker overshoots)",
        f"  Still failing after geodesic stitching: {f['geo_fail']}",
        "",
        "Each later page has 4 failure cases, 3 panels per row:",
        "  (1) sampled ridge H + walker trajectory (x = stuck point)",
        "  (2) break diagnosis: part of H >= 0.5 connected to start (green) / to goal (blue)",
        "  (3) stitched geodesic: blue = on ridge (H >= 0.5), red = bridge across a gap",
        "  dark gray = obstacles, light gray = 1px padding (area the planner actually avoids)",
    ]
    for i, s in enumerate(diag):
        fig.text(0.07, 0.40 - 0.026 * i, s, fontsize=9, weight="bold" if i == 0 else None)
    pdf.savefig(fig)
    plt.close(fig)


def case_page(pdf, cases, title):
    fig, axes = plt.subplots(len(cases), 3, figsize=(8.27, 2.75 * len(cases) + 0.6))
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=12, weight="bold")
    for row, c in zip(axes, cases, strict=False):
        w, h = c["world"], c["h"]
        base_axes(row[0], w, h)
        p = c["walk"]
        row[0].plot(p[:, 0], p[:, 1], "-", color=WALK, lw=1.8, zorder=5)
        if not c["walk_ok"]:
            row[0].plot(*p[-1], "x", color=WALK, ms=11, mew=3, zorder=6)
        row[0].set_title(f"map #{c['seed']}  walker {'ok' if c['walk_ok'] else 'fail'}", fontsize=9)

        comp = np.zeros(h.shape)
        comp[c["from_start"]] = 1
        comp[c["from_goal"]] = 2
        comp[c["from_start"] & c["from_goal"]] = 3
        base_axes(
            row[1],
            w,
            comp,
            cmap=ListedColormap(["#111111", "#22c55e", "#3b82f6", "#eab308"]),
            vmax=3,
        )
        row[1].set_title(f"Break diagnosis  b* = {c['b']:.3f}", fontsize=9)

        base_axes(row[2], w, h)
        g = c["geo"]
        if g is not None:
            px = c["geo_px"]
            on = h[px[:, 0].astype(int), px[:, 1].astype(int)] >= ON_RIDGE
            for k in range(len(g) - 1):
                row[2].plot(
                    g[k : k + 2, 0],
                    g[k : k + 2, 1],
                    "-",
                    lw=2.2,
                    zorder=5,
                    color=GEO
                    if on[min(k, len(on) - 1)] and on[min(k + 1, len(on) - 1)]
                    else BRIDGE,
                )
            gap = int((~on).sum())
            row[2].set_title(
                f"Geodesic {'ok' if c['geo_ok'] else 'fail'}  bridge {gap} cells", fontsize=9
            )
        else:
            row[2].set_title("Geodesic: unreachable", fontsize=9)
    for row in axes[len(cases) :]:
        for ax in row:
            ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ridgeflow_rrt64.pt")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--output", default="outputs/topology_report.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    model = RidgeModel(args.checkpoint, grid, device="cuda")
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]

    cases, successes = [], []
    n_ok = n_geo = n_either = 0
    wallhug = 0
    for begin in range(0, len(worlds), args.batch):
        chunk = worlds[begin : begin + args.batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si = torch.floor(starts / grid.cell).long()
        gi = torch.floor(goals / grid.cell).long()
        h = model.sample(occ, starts, goals, seed=begin)
        res = walker.walk(h, grids, starts, goals)
        b = bottleneck(h, ~grids, si, gi).cpu().numpy()
        b_raw = bottleneck(h, ~occ, si, gi).cpu().numpy()
        ridge = (h >= ON_RIDGE) & ~grids
        fs = reach(ridge, si).cpu().numpy()
        fg = reach(ridge, gi).cpu().numpy()
        paths = geodesic(h, ~grids, si, gi)
        hn = h.cpu().numpy()
        for k, w in enumerate(chunk):
            ok = bool(res.reached[k]) and w.path_free(res.paths[k])
            geo = None if paths[k] is None else geodesic_world(paths[k], w, grid.cell)
            geo_ok = geo is not None and w.path_free(geo)
            n_ok += ok
            n_geo += geo_ok
            n_either += ok or geo_ok
            c = dict(
                seed=begin + k,
                world=w,
                h=hn[k],
                walk=res.paths[k],
                walk_ok=ok,
                b=float(b[k]),
                from_start=fs[k],
                from_goal=fg[k],
                geo=geo,
                geo_px=paths[k],
                geo_ok=geo_ok,
            )
            if not ok:
                cases.append(c)
                wallhug += b[k] < 0.05 and b_raw[k] >= 0.5
            elif len(successes) < 4 and begin + k >= 0:
                successes.append(c)

    n = len(worlds)
    stats = {
        "rates": {
            "ridge walker (baseline)": 100 * n_ok / n,
            "geodesic as the path": 100 * n_geo / n,
            "stitch only when walker fails": 100 * n_either / n,
        },
        "fail": {
            "n": len(cases),
            "broken": sum(c["b"] < 0.05 for c in cases),
            "wallhug": int(wallhug),
            "geo_fail": sum(not c["geo_ok"] for c in cases),
        },
    }
    cases.sort(key=lambda c: (c["geo_ok"], c["b"] >= 0.05))  # unrepaired first, then connected
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        summary_page(pdf, stats, n)
        for i in range(0, len(cases), 4):
            case_page(
                pdf,
                cases[i : i + 4],
                f"Walker failures {i + 1}–{min(i + 4, len(cases))} / {len(cases)}",
            )
        case_page(pdf, successes, "Control: maps where walker succeeds (geodesic ~ walker)")
    print(stats)
    print(out)


if __name__ == "__main__":
    main()
