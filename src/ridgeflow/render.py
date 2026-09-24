from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ridgeflow.world import World

WALKER = "#00e68a"
DWA = "#ff4d4d"


@dataclass
class Panel:
    title: str
    world: World
    heatmap: np.ndarray | None = None
    walker: tuple[np.ndarray, bool] | None = None
    dwa: tuple[np.ndarray, bool] | None = None


def _draw(ax, panel: Panel) -> None:
    world = panel.world
    extent = world.grid.extent
    bounds = [0.0, extent, 0.0, extent]
    background = np.zeros((2, 2)) if panel.heatmap is None else panel.heatmap.T
    ax.imshow(background, cmap="magma", origin="lower", extent=bounds, vmin=0.0, vmax=1.0)
    occupancy = world.occupancy().numpy().T
    ax.imshow(
        np.ma.masked_where(~occupancy, occupancy), cmap="Greys", origin="lower",
        extent=bounds, vmin=0.0, vmax=1.4, alpha=0.9,
    )
    tracks = ((panel.walker, WALKER, "ridge walker", 5), (panel.dwa, DWA, "DWA", 4))
    for track, colour, label, z in tracks:
        if track is None:
            continue
        path, ok = track
        ax.plot(path[:, 0], path[:, 1], "-", color=colour, lw=2.2, label=label, zorder=z)
        if not ok:
            ax.plot(*path[-1], "x", color=colour, ms=12, mew=3, zorder=6)
    ax.plot(*world.start, "o", color="white", ms=9, mec="k", zorder=7)
    ax.plot(*world.goal, "*", color="cyan", ms=16, mec="k", zorder=7)
    ax.set_xlim(0.0, extent)
    ax.set_ylim(0.0, extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(panel.title, fontsize=11)


def figure(panels: list[Panel], output: str | Path, columns: int | None = None) -> Path:
    columns = columns or len(panels)
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.7 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, panels, strict=False):
        _draw(ax, panel)
    for ax in axes[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.05 if handles else 0, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
