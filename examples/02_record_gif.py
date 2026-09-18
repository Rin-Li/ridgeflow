"""Record episodes as two-panel animations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ridgeflow import FlowMatchingGuidance, GridSpec, SimulationConfig, sample_pocket_world
from ridgeflow.render import record, save_gif

DEFAULT_CHECKPOINT = "checkpoints/ridgeflow_rrt64.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--obstacle-speed", type=float, default=1.0)
    parser.add_argument("--guidance-hz", type=float, default=16.0)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    grid = GridSpec()
    guidance = FlowMatchingGuidance(args.checkpoint, grid, device=args.device)
    config = SimulationConfig(guidance_hz=args.guidance_hz)

    for seed in args.seeds:
        world = sample_pocket_world(
            np.random.default_rng(seed), grid, speed=args.obstacle_speed
        )
        result, frames = record(world, guidance, config, sample_seed=1_000_003 * (seed + 1))
        path = Path(args.outdir) / f"pocket_speed{args.obstacle_speed:g}_seed{seed}.gif"
        save_gif(
            result,
            frames,
            path,
            extent=grid.extent,
            fps=args.fps,
            stride=args.stride,
            title=f"obstacle speed {args.obstacle_speed:g} | ",
        )
        print(f"{path}  ({result.outcome.value}, {len(frames)} frames)")


if __name__ == "__main__":
    main()
