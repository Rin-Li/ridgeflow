"""One PDF over the ablation JSONs: TMP64 components, TMP64 sigma sweep, RRT ablations.

Reads (any that exist) outputs/ablation/{tmp64_components,tmp64_sigma,rrt}.json, written by
tmp64_eval.py --json and rrt_ablation.py --json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

GREY, ORANGE, BLUE = "#9a9a9a", "#ff8c1a", "#3366cc"


def load(folder: Path, name: str):
    p = folder / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def table(ax, cells, labels, size=9):
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=labels, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(size)
    tab.scale(1, 1.5)


def bars(ax, names, values, title, fmt, highlight=0, err=None):
    colours = [ORANGE if i == highlight else GREY for i in range(len(names))]
    ax.bar(names, values, color=colours, yerr=err, capsize=3)
    lo, hi = min(values), max(values)
    pad = max(hi - lo, 1e-3) * 0.6
    ax.set_ylim(lo - pad, hi + pad * 0.8)
    for i, v in enumerate(values):
        ax.text(i, v + pad * 0.08, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def tmp64_components(pdf, rows):
    k1 = [r for r in rows if r["mode"] == "K1"]
    b8 = {r["model"]: r for r in rows if r["mode"] == "best-of-8"}
    names = [r["model"] for r in k1]
    full = k1[0]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(
        "TMP64 component ablation (all trained from scratch, 300 epochs; "
        "full = gap loss + augmentation + geodesic target, σ1.5)",
        fontsize=13,
        weight="bold",
    )
    specs = [
        ("conn", "Single-draw connectivity %", "{:.1f}", None),
        ("red", "Single-draw expansion reduction % (±SE)", "{:.1f}", "red_sd"),
        ("cost", "Single-draw cost ratio", "{:.3f}", None),
        ("cross", "Wall crossing %", "{:.1f}", None),
    ]
    for i, (key, title, fmt, err) in enumerate(specs):
        ax = fig.add_axes([0.06 + 0.235 * i, 0.5, 0.19, 0.36])
        bars(ax, names, [r[key] for r in k1], title, fmt, err=[r[err] for r in k1] if err else None)
    cells = []
    for r in k1:
        o = b8.get(r["model"], {})
        cells.append(
            [
                r["model"],
                f"{r['conn']:.1f}%",
                f"{r['conn'] - full['conn']:+.1f}",
                f"{r['red']:.1f} ± {r['red_sd']:.1f}",
                f"{r['red'] - full['red']:+.1f}",
                f"{r['cost']:.3f}",
                f"{r['cross']:.1f}%",
                f"{o.get('conn', float('nan')):.1f}%",
                f"{o.get('red', float('nan')):.1f}",
            ]
        )
    labels = [
        "Model",
        "Conn.",
        "Δ conn.",
        "Exp. red. %",
        "Δ red.",
        "Cost ratio",
        "Wall X",
        "Best-of-8 conn.",
        "Best-of-8 red.",
    ]
    table(fig.add_axes([0.04, 0.03, 0.92, 0.36]), cells, labels)
    pdf.savefig(fig)
    plt.close(fig)


def tmp64_sigma(pdf, rows):
    sig = lambda name: float(name.replace("σ", ""))  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.suptitle(
        "TMP64 σ sweep (full recipe, 300 epochs from scratch; σ0 = no geodesic target)",
        fontsize=13,
        weight="bold",
    )
    for mode, colour in (("K1", ORANGE), ("best-of-8", BLUE)):
        rs = sorted((r for r in rows if r["mode"] == mode), key=lambda r: sig(r["model"]))
        x = [sig(r["model"]) for r in rs]
        axes[0].plot(x, [r["conn"] for r in rs], "o-", color=colour, label=mode)
        axes[1].errorbar(
            x, [r["red"] for r in rs], [r["red_sd"] for r in rs], fmt="o-", color=colour, label=mode
        )
        axes[2].plot(x, [r["cost"] for r in rs], "o-", color=colour, label=mode)
    for ax, t in zip(axes, ("Connectivity %", "Expansion reduction %", "Cost ratio"), strict=True):
        ax.set_xlabel("σ (pixels)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.42, 1, 0.93))
    cells = [
        [
            r["model"],
            r["mode"],
            f"{r['conn']:.1f}%",
            f"{r['cross']:.1f}%",
            f"{r['red']:.1f} ± {r['red_sd']:.1f}",
            f"{r['cost']:.3f}",
            f"{r['gen_ms']:.2f}",
        ]
        for r in sorted(rows, key=lambda r: (r["mode"] != "K1", sig(r["model"])))
        if r["mode"] in ("K1", "best-of-8")
    ]
    labels = ["σ", "Draws", "Conn.", "Wall X", "Exp. red. %", "Cost ratio", "Gen ms"]
    table(fig.add_axes([0.1, 0.02, 0.8, 0.36]), cells, labels, size=8)
    pdf.savefig(fig)
    plt.close(fig)


def rrt(pdf, rows):
    names = [r["model"] for r in rows]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(
        "RRT (continuous) component ablation: 1000 maps × 4 draws", fontsize=13, weight="bold"
    )
    specs = [
        ("connected", "Single-draw connectivity %", "{:.1f}"),
        ("raw_dfs", "No projection, backtracking walker succ. %", "{:.1f}"),
        ("proj_dfs", "Projection, backtracking walker succ. %", "{:.2f}"),
        ("diversity", "Diversity (1−IoU)", "{:.3f}"),
    ]
    for i, (key, title, fmt) in enumerate(specs):
        ax = fig.add_axes([0.06 + 0.235 * i, 0.5, 0.19, 0.36])
        bars(ax, names, [r[key] for r in rows], title, fmt)
    cells = [
        [
            r["model"],
            f"{r['connected']:.1f}%",
            f"{r['walls']:.1f}%",
            f"{r['gap_cells']:.2f}",
            f"{r['hedging']:.1f}%",
            f"{r['diversity']:.3f}",
            f"{r['raw_plain']:.1f}%",
            f"{r['raw_dfs']:.1f}%",
            f"{r['proj_dfs']:.2f}%",
            f"{r['turning']:.2f}",
        ]
        for r in rows
    ]
    labels = [
        "Model",
        "Conn.",
        "Wall X",
        "Gap cells",
        "Two-way",
        "Diversity",
        "Raw·plain",
        "Raw·DFS",
        "Proj·DFS",
        "Total turn",
    ]
    table(fig.add_axes([0.03, 0.03, 0.94, 0.36]), cells, labels, size=8)
    pdf.savefig(fig)
    plt.close(fig)

    # inference pipeline on the full model: each step added in turn
    r = rows[0]
    steps = [
        ("Single-draw heatmap\n(connectivity)", r["connected"]),
        ("+ plain walker", r["raw_plain"]),
        ("Swap in backtracking walker", r["raw_dfs"]),
        ("Projection + plain walker", r["proj_plain"]),
        ("Projection + backtracking walker", r["proj_dfs"]),
    ]
    fig, ax = plt.subplots(figsize=(11.69, 6))
    fig.suptitle(
        f"RRT inference-pipeline ablation ({r['model']} model): success rate %",
        fontsize=13,
        weight="bold",
    )
    bars(ax, [s for s, _ in steps], [v for _, v in steps], "", "{:.2f}", highlight=len(steps) - 1)
    ax.text(
        0.01,
        -0.18,
        f"Per query: generate {r['gen_ms']:.2f} ms, projection {r['proj_ms']:.2f} ms, "
        f"backtracking walker {r['walk_ms']:.2f} ms (batch-amortized). "
        "Success = reached and collision-free.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="outputs/ablation")
    parser.add_argument("--output", default="outputs/ablation/ablation_summary.pdf")
    args = parser.parse_args()
    folder = Path(args.folder)
    with PdfPages(args.output) as pdf:
        for name, draw in (
            ("tmp64_components", tmp64_components),
            ("tmp64_sigma", tmp64_sigma),
            ("rrt", rrt),
        ):
            rows = load(folder, name)
            if rows:
                draw(pdf, rows)
            else:
                print(f"skip {name}: no json")
    print(args.output)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
