"""Sweep obstacle speed and print the success table from the README."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ridgeflow import (
    FlowMatchingGuidance,
    GridSpec,
    SimulationConfig,
    run_episode,
    sample_pocket_world,
    sample_scattered_world,
    summarise,
)

DEFAULT_CHECKPOINT = "checkpoints/ridgeflow_rrt64.pt"
SAMPLERS = {"pocket": sample_pocket_world, "scattered": sample_scattered_world}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scenario", default="pocket", choices=sorted(SAMPLERS))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--movers", type=int, default=4, help="pocket scenario only")
    parser.add_argument(
        "--obstacle-speed", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0]
    )
    parser.add_argument("--guidance-hz", type=float, default=16.0)
    parser.add_argument("--control-hz", type=float, default=16.0)
    parser.add_argument("--robot-speed", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="optional csv of every episode")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    grid = GridSpec()
    guidance = FlowMatchingGuidance(args.checkpoint, grid, device=args.device)
    sampler = SAMPLERS[args.scenario]
    config = SimulationConfig(
        control_hz=args.control_hz,
        guidance_hz=args.guidance_hz,
        robot_speed=args.robot_speed,
    )

    extra = {"movers": args.movers} if args.scenario == "pocket" else {}

    rows = []
    for speed in args.obstacle_speed:
        results = []
        for seed in range(args.episodes):
            world = sampler(np.random.default_rng(seed), grid, speed=speed, **extra)
            result = run_episode(world, guidance, config, sample_seed=1_000_003 * (seed + 1))
            results.append(result)
            rows.append(
                asdict(result)
                | {"outcome": result.outcome.value, "seed": seed, "obstacle_speed": speed}
            )
        summary = summarise(results)
        print(summary.row(f"v_obstacle={speed:<5g}"), flush=True)

    print(f"\n{guidance.calls} inferences at {guidance.ms_per_call:.2f} ms each")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
