from __future__ import annotations

import numpy as np
import pytest

from ridgeflow import GridSpec, Outcome, SimulationConfig, World, run_episode, summarise
from ridgeflow.scenarios import sample_pocket_world


def test_step_size_follows_speed_and_rate(grid: GridSpec) -> None:
    config = SimulationConfig(control_hz=16.0, robot_speed=1.0)
    assert config.step_px(grid) == pytest.approx(0.5)
    assert SimulationConfig(control_hz=32.0).step_px(grid) == pytest.approx(0.25)


def test_refresh_interval_matches_the_rate_ratio() -> None:
    assert SimulationConfig(control_hz=16.0, guidance_hz=16.0).refresh_every() == 1
    assert SimulationConfig(control_hz=16.0, guidance_hz=4.0).refresh_every() == 4
    assert SimulationConfig(guidance_hz=0.0).refresh_every() is None


def test_episode_reaches_the_goal_on_an_empty_map(grid: GridSpec, guidance) -> None:
    world = World([], np.array([1.0, 1.0]), np.array([7.0, 7.0]), grid)
    result = run_episode(world, guidance, SimulationConfig())
    assert result.outcome is Outcome.ARRIVED
    assert result.path_length == pytest.approx(np.sqrt(2) * 6.0, rel=0.1)


def test_replanning_costs_one_inference_per_interval(grid: GridSpec, guidance) -> None:
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    result = run_episode(world, guidance, SimulationConfig(control_hz=16.0, guidance_hz=4.0))
    assert result.inferences == pytest.approx(1 + result.steps // 4, abs=2)


def test_planning_once_makes_a_single_inference(grid: GridSpec, guidance) -> None:
    world = World([], np.array([1.0, 4.0]), np.array([7.0, 4.0]), grid)
    result = run_episode(world, guidance, SimulationConfig(guidance_hz=0.0))
    assert result.inferences == 1
    assert np.isnan(result.staleness_px)


def test_staleness_reports_pixels_travelled_between_replans(grid: GridSpec, guidance) -> None:
    world = sample_pocket_world(np.random.default_rng(0), grid, speed=2.0)
    result = run_episode(world, guidance, SimulationConfig(guidance_hz=8.0))
    assert result.staleness_px == pytest.approx(2.0 / 8.0 / grid.cell)


def test_summary_rates_sum_to_one(grid: GridSpec, guidance) -> None:
    worlds = [sample_pocket_world(np.random.default_rng(s), grid, speed=1.0) for s in range(3)]
    results = [run_episode(w, guidance, SimulationConfig()) for w in worlds]
    summary = summarise(results)
    assert sum(summary.rates.values()) == pytest.approx(1.0)
    assert summary.episodes == 3
