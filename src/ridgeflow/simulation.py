from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ridgeflow.grid import GridSpec
from ridgeflow.guidance import Guidance
from ridgeflow.walker import RidgeWalker, StepOutcome, WalkerConfig
from ridgeflow.world import World


class Outcome(str, Enum):
    ARRIVED = "arrived"
    HIT = "hit"
    TRAPPED = "trapped"
    FROZEN = "frozen"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SimulationConfig:
    """Rates and world dynamics. ``guidance_hz = 0`` plans once and never replans.

    ``latency_s`` is the inference delay the lookahead assumes. ``None`` uses the running
    mean the guidance measures, which is physically faithful but makes a run depend on
    machine timing; a fixed value keeps benchmarks reproducible.
    """

    control_hz: float = 16.0
    guidance_hz: float = 16.0
    robot_speed: float = 1.0
    timeout_s: float = 30.0
    latency_s: float | None = 0.011
    freeze_limit: int = 160
    avoid_radius: float = 1.5
    avoid_gain: float = 2.0
    goal_bay_half: float = 0.45

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_steps(self) -> int:
        return int(self.timeout_s * self.control_hz)

    def step_px(self, grid: GridSpec) -> float:
        return self.robot_speed / self.control_hz / grid.cell

    def refresh_every(self) -> int | None:
        if self.guidance_hz <= 0.0:
            return None
        return max(1, int(round(self.control_hz / self.guidance_hz)))


@dataclass
class EpisodeResult:
    outcome: Outcome
    steps: int
    seconds: float
    path_length: float
    min_clearance: float
    inferences: int
    waited: int
    staleness_px: float

    @property
    def arrived(self) -> bool:
        return self.outcome is Outcome.ARRIVED


StepHook = Callable[[int, World, RidgeWalker, bool], None]


def _anchor(world: World, walker: RidgeWalker, config: SimulationConfig, latency_s: float):
    """Where the robot will be when this field starts being used, not where it is now.

    Inference takes time and the result then has to serve until the next refresh, so the
    pose that matters is the one at the end of that interval. Anchoring on the robot
    itself also puts the field's bright start cap directly under it, which flattens the
    score across the whole candidate fan.
    """
    horizon = 1.0 / config.guidance_hz if config.guidance_hz > 0.0 else 0.0
    lookahead = config.robot_speed * (horizon + latency_s)
    ahead = world.grid.clip(walker.position_world + walker.heading * lookahead)
    return walker.position_world if world.blocked(ahead) else ahead


def run_episode(
    world: World,
    guidance: Guidance,
    config: SimulationConfig | None = None,
    walker_config: WalkerConfig | None = None,
    sample_seed: int = 0,
    on_step: StepHook | None = None,
) -> EpisodeResult:
    """Drive one robot from start to goal while the boxes move underneath it.

    A single sampler seed is held for the whole episode: drawing a fresh one per refresh
    makes the stochastic sampler propose a different route every cycle, so the robot
    would chase a new plan instead of tracking the map.
    """
    config = config or SimulationConfig()
    refresh_every = config.refresh_every()
    latency_s = config.latency_s

    heatmap = guidance.heatmap(world.occupancy(), world.start, world.goal, sample_seed)
    walker = RidgeWalker(
        heatmap,
        world.start,
        world.goal,
        step_px=config.step_px(world.grid),
        grid=world.grid,
        config=walker_config,
    )

    inferences = 1
    consecutive_waits = 0
    total_waits = 0
    min_clearance = np.inf
    outcome = Outcome.TIMEOUT

    for step in range(config.max_steps):
        refreshed = refresh_every is not None and step > 0 and step % refresh_every == 0
        if refreshed:
            if config.latency_s is None:
                latency_s = guidance.ms_per_call / 1000.0
            anchor = _anchor(world, walker, config, latency_s)
            walker.set_heatmap(
                guidance.heatmap(world.occupancy(), anchor, world.goal, sample_seed)
            )
            inferences += 1

        if on_step is not None:
            on_step(step, world, walker, refreshed)

        status = walker.step(world.occupancy())
        if status is StepOutcome.ARRIVED:
            outcome = Outcome.ARRIVED
            break
        if status is StepOutcome.TRAPPED:
            outcome = Outcome.TRAPPED
            break
        if status is StepOutcome.WAITED:
            consecutive_waits += 1
            total_waits += 1
            if consecutive_waits >= config.freeze_limit:
                outcome = Outcome.FROZEN
                break
        else:
            consecutive_waits = 0

        world.advance(
            config.dt,
            robot=walker.position_world,
            avoid_radius=config.avoid_radius,
            avoid_gain=config.avoid_gain,
            keep_clear=(world.goal,),
            bay_half=config.goal_bay_half,
        )
        min_clearance = min(min_clearance, world.clearance(walker.position_world))
        if world.blocked(walker.position_world):
            outcome = Outcome.HIT
            break

    if on_step is not None:
        on_step(-1, world, walker, False)

    trail = walker.trail_world
    obstacle_speed = max((rect.speed for rect in world.rects), default=0.0)
    staleness = np.nan
    if config.guidance_hz > 0.0:
        staleness = obstacle_speed / config.guidance_hz / world.grid.cell
    return EpisodeResult(
        outcome=outcome,
        steps=len(walker.trail) - 1,
        seconds=world.time,
        path_length=float(np.linalg.norm(np.diff(trail, axis=0), axis=1).sum()),
        min_clearance=float(min_clearance if np.isfinite(min_clearance) else 0.0),
        inferences=inferences,
        waited=total_waits,
        staleness_px=float(staleness),
    )
