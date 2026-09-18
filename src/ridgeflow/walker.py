from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ridgeflow.field import bilinear_sample, squared_distance_field
from ridgeflow.grid import GridSpec

_PROBE_RADIUS = 0.1
_PROBE = np.array(
    [
        [0.0, 0.0],
        [_PROBE_RADIUS, 0.0],
        [-_PROBE_RADIUS, 0.0],
        [0.0, _PROBE_RADIUS],
        [0.0, -_PROBE_RADIUS],
        [_PROBE_RADIUS, _PROBE_RADIUS],
        [-_PROBE_RADIUS, _PROBE_RADIUS],
        [_PROBE_RADIUS, -_PROBE_RADIUS],
        [-_PROBE_RADIUS, -_PROBE_RADIUS],
    ]
)


class StepOutcome(str, Enum):
    MOVED = "moved"
    WAITED = "waited"
    ARRIVED = "arrived"
    TRAPPED = "trapped"


@dataclass(frozen=True)
class WalkerConfig:
    """Dynamic window and scoring. The published results use these values unchanged."""

    cone_deg: float = 75.0
    turn_samples: int = 21
    speeds: tuple[float, ...] = (1.0, 0.0, -0.5)
    ridge_sigma: float = 1.5
    blur_sigma: float = 1.5
    goal_weight: float = 0.0
    forward_bonus: float = 0.0
    goal_tolerance_px: float = 2.0
    collision_samples: int = 5


class RidgeWalker:
    """A dynamic-window controller whose only objective is to descend a ridge field.

    Each cycle it enumerates the (turn, speed) product, discards the candidates whose
    swept segment touches an occupied cell, and takes the survivor closest to the ridge.
    Speed zero rotates on the spot and negative speed reverses, so a dead end is
    escapable. The field is replaced whenever new guidance arrives; nothing else about
    the controller changes between cycles.
    """

    def __init__(
        self,
        heatmap: np.ndarray,
        start_world,
        goal_world,
        step_px: float,
        grid: GridSpec | None = None,
        config: WalkerConfig | None = None,
    ) -> None:
        self.grid = grid or GridSpec()
        self.config = config or WalkerConfig()
        self.step_px = float(step_px)
        self.position = self.grid.to_pixel(start_world)
        self.goal = self.grid.to_pixel(goal_world)
        self.trail = [self.position.copy()]
        heading = self.goal - self.position
        self.heading = heading / max(np.linalg.norm(heading), 1e-9)
        self.set_heatmap(heatmap)

    def set_heatmap(self, heatmap: np.ndarray) -> None:
        self.heatmap = np.asarray(heatmap, np.float64)
        self.distance_field = squared_distance_field(
            self.heatmap, self.config.ridge_sigma, self.config.blur_sigma
        )

    @property
    def position_world(self) -> np.ndarray:
        return self.grid.to_world(self.position)

    @property
    def trail_world(self) -> np.ndarray:
        return self.grid.to_world(np.asarray(self.trail))

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def candidates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The dynamic window: every turn crossed with every speed."""
        angles = np.radians(
            np.linspace(-self.config.cone_deg, self.config.cone_deg, self.config.turn_samples)
        )
        cos, sin = np.cos(angles), np.sin(angles)
        headings = np.stack(
            [
                self.heading[0] * cos - self.heading[1] * sin,
                self.heading[1] * cos + self.heading[0] * sin,
            ],
            axis=1,
        )
        speeds = np.repeat(np.asarray(self.config.speeds, np.float64), len(headings))
        headings = np.tile(headings, (len(self.config.speeds), 1))
        positions = self.position[None] + (speeds[:, None] * self.step_px) * headings
        return positions, headings, speeds

    def feasible(self, targets: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """Dense sample of each swept segment, dilated by the collision probe."""
        size = self.grid.size
        ts = np.linspace(0.0, 1.0, self.config.collision_samples)[None, :, None]
        swept = self.position[None, None] + ts * (targets[:, None] - self.position[None, None])
        probed = swept[:, :, None, :] + _PROBE[None, None]
        xs = np.clip(np.round(probed[..., 0]).astype(int), 0, size - 1)
        ys = np.clip(np.round(probed[..., 1]).astype(int), 0, size - 1)
        return ~(occupancy[xs, ys] > 0.5).any(axis=(1, 2))

    def score(self, positions: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """Normalised ridge proximity, plus the optional goal and velocity terms."""
        squared = bilinear_sample(self.distance_field, positions[:, 0], positions[:, 1])
        span = float(squared.max() - squared.min())
        ridge = (squared.max() - squared) / span if span > 1e-9 else np.zeros(len(squared))
        if self.config.goal_weight == 0.0 and self.config.forward_bonus == 0.0:
            return ridge
        progress = (
            self.distance_to_goal - np.linalg.norm(self.goal - positions, axis=1)
        ) / self.step_px
        return ridge + self.config.goal_weight * progress + self.config.forward_bonus * speeds

    def step(self, occupancy: np.ndarray) -> StepOutcome:
        """Advance one control cycle against the currently observed occupancy."""
        if self.distance_to_goal <= self.config.goal_tolerance_px:
            return StepOutcome.ARRIVED

        positions, headings, speeds = self.candidates()
        keep = self.feasible(positions, occupancy)
        if not keep.any():
            if not self.feasible(self.position[None], occupancy)[0]:
                return StepOutcome.TRAPPED
            return StepOutcome.WAITED
        positions, headings, speeds = positions[keep], headings[keep], speeds[keep]

        best = int(np.argmax(self.score(positions, speeds)))
        self.heading = headings[best]
        self.position = positions[best]
        if abs(speeds[best]) <= 1e-9:
            return StepOutcome.WAITED
        self.trail.append(self.position.copy())
        return StepOutcome.MOVED
