from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.patches import Rectangle

from ridgeflow.guidance import Guidance
from ridgeflow.simulation import EpisodeResult, SimulationConfig, run_episode
from ridgeflow.walker import RidgeWalker, WalkerConfig
from ridgeflow.world import World


@dataclass
class Frame:
    index: int
    time: float
    refreshed: bool
    rects: list[tuple[float, float, float, float, float, float]]
    robot: np.ndarray
    trail: np.ndarray
    heatmap: np.ndarray
    distance_field: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    fan: tuple[np.ndarray, np.ndarray] | None


def _fan(walker: RidgeWalker, world: World) -> tuple[np.ndarray, np.ndarray] | None:
    positions, _, speeds = walker.candidates()
    keep = walker.feasible(positions, world.occupancy())
    if not keep.any():
        return None
    positions, speeds = positions[keep], speeds[keep]
    scores = walker.score(positions, speeds)
    span = float(scores.max() - scores.min())
    normalised = (scores - scores.min()) / span if span > 1e-9 else np.zeros(len(scores))
    return walker.grid.to_world(positions), normalised


def record(
    world: World,
    guidance: Guidance,
    config: SimulationConfig | None = None,
    walker_config: WalkerConfig | None = None,
    sample_seed: int = 0,
) -> tuple[EpisodeResult, list[Frame]]:
    """Run an episode, keeping everything a frame needs."""
    frames: list[Frame] = []

    def capture(index: int, world: World, walker: RidgeWalker, refreshed: bool) -> None:
        frames.append(
            Frame(
                index=index,
                time=world.time,
                refreshed=refreshed,
                rects=[(r.x, r.y, r.width, r.height, r.vx, r.vy) for r in world.rects],
                robot=walker.position_world.copy(),
                trail=walker.trail_world.copy(),
                heatmap=walker.heatmap.copy(),
                distance_field=walker.distance_field.copy(),
                start=world.start.copy(),
                goal=world.goal.copy(),
                fan=_fan(walker, world),
            )
        )

    result = run_episode(world, guidance, config, walker_config, sample_seed, capture)
    return result, frames


def save_gif(
    result: EpisodeResult,
    frames: list[Frame],
    output: str | Path,
    extent: float = 8.0,
    fps: int = 12,
    stride: int = 2,
    title: str = "",
) -> Path:
    """Write the episode as a two-panel animation."""
    kept = frames[::stride] + ([frames[-1]] if stride > 1 else [])
    bounds = [0.0, extent, 0.0, extent]

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), dpi=90)
    labels = ("ridge field + world", "distance field + dynamic window")
    for axis, label in zip(axes, labels, strict=True):
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        axis.set_title(label, fontsize=10)

    ridge = axes[0].imshow(
        np.zeros((2, 2)), cmap="magma", origin="lower", extent=bounds, vmin=0.0, vmax=1.0, zorder=0
    )
    field = axes[1].imshow(
        np.zeros((2, 2)), cmap="viridis_r", origin="lower", extent=bounds, zorder=0
    )
    boxes: list[Rectangle] = []
    arrows: list = []
    rays: list = []
    trails = [axis.plot([], [], "-", color="#00ff9d", lw=2.0, zorder=6)[0] for axis in axes]
    robots = [axis.plot([], [], "o", color="w", ms=8, mec="k", zorder=8)[0] for axis in axes]
    goals = [axis.plot([], [], "*", color="cyan", ms=14, mec="k", zorder=7)[0] for axis in axes]
    origins = [axis.plot([], [], "o", color="lime", ms=7, mec="k", zorder=7)[0] for axis in axes]
    caption = figure.suptitle("", fontsize=11)

    def draw(position: int):
        frame = kept[position]
        for artists in (boxes, arrows, rays):
            for artist in artists:
                artist.remove()
            artists.clear()

        for x, y, width, height, vx, vy in frame.rects:
            for axis, alpha in ((axes[0], 0.55), (axes[1], 0.35)):
                patch = Rectangle(
                    (x, y), width, height, facecolor="#8fb3cc", edgecolor="w",
                    lw=0.8, alpha=alpha, zorder=3,
                )
                axis.add_patch(patch)
                boxes.append(patch)
            if abs(vx) + abs(vy) > 1e-9:
                arrows.append(
                    axes[0].arrow(
                        x + width / 2, y + height / 2, vx * 0.5, vy * 0.5,
                        head_width=0.16, color="#ffd400", zorder=4,
                    )
                )

        ridge.set_data(frame.heatmap.T)
        field.set_data(frame.distance_field.T)
        field.set_clim(0.0, float(np.percentile(frame.distance_field, 85)))

        if frame.fan is not None:
            points, scores = frame.fan
            for point, score in zip(points, scores, strict=False):
                line, = axes[1].plot(
                    [frame.robot[0], point[0]], [frame.robot[1], point[1]],
                    "-", color=cm.autumn(float(score)), lw=1.5, zorder=6,
                )
                rays.append(line)

        for artist in trails:
            artist.set_data(frame.trail[:, 0], frame.trail[:, 1])
        for artist in robots:
            artist.set_data([frame.robot[0]], [frame.robot[1]])
        for goal, origin in zip(goals, origins, strict=False):
            goal.set_data([frame.goal[0]], [frame.goal[1]])
            origin.set_data([frame.start[0]], [frame.start[1]])

        state = result.outcome.value if frame.index < 0 else "running"
        marker = "   <<< replanned" if frame.refreshed else ""
        caption.set_text(f"{title}t={frame.time:.2f}s | {state}{marker}")
        return []

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.FuncAnimation(figure, draw, frames=len(kept), interval=1000 / fps).save(
        output, writer=animation.PillowWriter(fps=fps)
    )
    plt.close(figure)
    return output
