"""Run one episode on a pocket map and print the outcome."""
from __future__ import annotations

import argparse

import numpy as np

from ridgeflow import (
    FlowMatchingGuidance,
    GridSpec,
    SimulationConfig,
    run_episode,
    sample_pocket_world,
)

DEFAULT_CHECKPOINT = "checkpoints/ridgeflow_rrt64.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacle-speed", type=float, default=1.0)
    parser.add_argument("--guidance-hz", type=float, default=16.0)
    parser.add_argument("--control-hz", type=float, default=16.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    grid = GridSpec()
    guidance = FlowMatchingGuidance(args.checkpoint, grid, device=args.device)
    world = sample_pocket_world(np.random.default_rng(args.seed), grid, speed=args.obstacle_speed)
    config = SimulationConfig(control_hz=args.control_hz, guidance_hz=args.guidance_hz)

    result = run_episode(world, guidance, config, sample_seed=1_000_003 * (args.seed + 1))

    print(f"outcome        {result.outcome.value}")
    print(f"path length    {result.path_length:.2f} world units")
    print(f"duration       {result.seconds:.2f} s over {result.steps} control steps")
    print(f"min clearance  {result.min_clearance:.2f} world units")
    print(f"inferences     {result.inferences} at {guidance.ms_per_call:.2f} ms each")
    print(f"staleness      {result.staleness_px:.2f} px between replans")


if __name__ == "__main__":
    main()
