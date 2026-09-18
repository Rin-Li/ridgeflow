from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ridgeflow.grid import GridSpec


@dataclass
class Rect:
    """Axis-aligned box with a constant-speed velocity, anchored at its lower-left corner."""

    x: float
    y: float
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x + self.width / 2.0, self.y + self.height / 2.0])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def advance(self, dt: float, extent: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 0.0:
            self.x, self.vx = 0.0, abs(self.vx)
        if self.x + self.width > extent:
            self.x, self.vx = extent - self.width, -abs(self.vx)
        if self.y < 0.0:
            self.y, self.vy = 0.0, abs(self.vy)
        if self.y + self.height > extent:
            self.y, self.vy = extent - self.height, -abs(self.vy)

    def overlaps(self, other: Rect, pad: float = 0.0) -> bool:
        return not (
            self.x + self.width + pad <= other.x
            or other.x + other.width + pad <= self.x
            or self.y + self.height + pad <= other.y
            or other.y + other.height + pad <= self.y
        )

    def evict_from(self, centre, half: float) -> bool:
        """Push this box out of the square of half-width ``half`` around ``centre``."""
        low = (centre[0] - half, centre[1] - half)
        high = (centre[0] + half, centre[1] + half)
        if (
            self.x + self.width <= low[0]
            or high[0] <= self.x
            or self.y + self.height <= low[1]
            or high[1] <= self.y
        ):
            return False
        right = high[0] - self.x
        left = (self.x + self.width) - low[0]
        up = high[1] - self.y
        down = (self.y + self.height) - low[1]
        shortest = min(right, left, up, down)
        if shortest == right:
            self.x, self.vx = self.x + right, abs(self.vx)
        elif shortest == left:
            self.x, self.vx = self.x - left, -abs(self.vx)
        elif shortest == up:
            self.y, self.vy = self.y + up, abs(self.vy)
        else:
            self.y, self.vy = self.y - down, -abs(self.vy)
        return True


def any_overlap(candidate: Rect, others: list[Rect], pad: float = 0.0) -> bool:
    return any(candidate.overlaps(other, pad) for other in others)


@dataclass
class World:
    """Moving rectangles, a start and a goal, rasterised onto a fixed grid."""

    rects: list[Rect]
    start: np.ndarray
    goal: np.ndarray
    grid: GridSpec = field(default_factory=GridSpec)
    time: float = 0.0
    _occupancy: np.ndarray | None = field(default=None, repr=False, compare=False)

    def advance(
        self,
        dt: float,
        robot=None,
        avoid_radius: float = 1.5,
        avoid_gain: float = 2.0,
        keep_clear: tuple = (),
        bay_half: float = 0.45,
    ) -> None:
        """Move every box, bending headings away from the robot and the reserved points."""
        repellers = ([] if robot is None else [np.asarray(robot, np.float64)]) + [
            np.asarray(point, np.float64) for point in keep_clear
        ]
        for rect in self.rects:
            if repellers and avoid_gain > 0.0:
                self._steer(rect, repellers, avoid_radius, avoid_gain)
            rect.advance(dt, self.grid.extent)
            for point in keep_clear:
                rect.evict_from(point, bay_half)
        self.time += dt
        self._occupancy = None

    @staticmethod
    def _steer(rect: Rect, repellers, radius: float, gain: float) -> None:
        speed = rect.speed
        if speed <= 1e-9:
            return
        centre = rect.centre
        vx, vy = rect.vx, rect.vy
        for target in repellers:
            offset = centre - target
            distance = float(np.linalg.norm(offset))
            if 1e-6 < distance < radius:
                weight = gain * (1.0 - distance / radius) * speed / distance
                vx += weight * offset[0]
                vy += weight * offset[1]
        norm = float(np.hypot(vx, vy))
        if norm > 1e-9:
            rect.vx, rect.vy = speed * vx / norm, speed * vy / norm

    def occupancy(self) -> np.ndarray:
        """Conservative ``[x, y]`` raster: every cell a rectangle touches is blocked."""
        if self._occupancy is None:
            cell = self.grid.cell
            size = self.grid.size
            grid = np.zeros((size, size), np.float32)
            for rect in self.rects:
                x0 = max(int(np.floor(rect.x / cell)), 0)
                x1 = min(int(np.ceil((rect.x + rect.width) / cell)), size)
                y0 = max(int(np.floor(rect.y / cell)), 0)
                y1 = min(int(np.ceil((rect.y + rect.height) / cell)), size)
                grid[x0:x1, y0:y1] = 1.0
            self._occupancy = grid
        return self._occupancy

    def blocked(self, point) -> bool:
        """Collision against the same raster the planner is shown."""
        index = self.grid.index(point)
        if not self.grid.holds(index):
            return True
        return bool(self.occupancy()[index] > 0.5)

    def segment_blocked(self, a, b, samples: int = 200) -> bool:
        ts = np.linspace(0.0, 1.0, samples)[:, None]
        points = np.asarray(a)[None] + ts * (np.asarray(b) - np.asarray(a))[None]
        return any(self.blocked(point) for point in points)

    def clearance(self, point) -> float:
        """Distance to the nearest rectangle; zero inside one."""
        best = np.inf
        for rect in self.rects:
            dx = max(rect.x - point[0], 0.0, point[0] - (rect.x + rect.width))
            dy = max(rect.y - point[1], 0.0, point[1] - (rect.y + rect.height))
            best = min(best, float(np.hypot(dx, dy)))
        return float(best) if np.isfinite(best) else 0.0
