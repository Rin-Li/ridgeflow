"""Numba kernel for the ridge walker: the same algorithm as ``RidgeWalker``'s torch loop,
step for step, compiled to machine code and run on the CPU, one thread per query.

The torch loop is launch-bound (about 130 tiny kernels and several host syncs per step); this
kernel does the same ~10k scalar operations per step without any of that overhead.
"""

from __future__ import annotations

import math
import os

import numba
import numpy as np
from numba import njit, prange

if "NUMBA_THREADING_LAYER" not in os.environ:
    # workqueue: no thread-pool wake-up per call (OpenMP costs ~30 ms for a single query here),
    # and TBB on many systems is too old and only warns
    numba.config.THREADING_LAYER = "workqueue"

ACTIVE, ARRIVED, STUCK = 0, 1, 2
_JITTER = 0.1 * np.array(
    [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
        [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5],
    ]
)  # fmt: skip


@njit(cache=True)
def _blocked(grid, px, py, cell, jitter):
    """Same rule as ``collision.blocked``: off-grid, the cell itself, or any jitter probe."""
    s = grid.shape[0]
    ix = int(math.floor(px / cell))
    iy = int(math.floor(py / cell))
    if ix < 0 or ix >= s or iy < 0 or iy >= s:
        return True
    if grid[ix, iy]:
        return True
    for k in range(jitter.shape[0]):
        jx = int(math.floor((px + jitter[k, 0] * cell) / cell))
        jy = int(math.floor((py + jitter[k, 1] * cell) / cell))
        if 0 <= jx < s and 0 <= jy < s and grid[jx, jy]:
            return True
    return False


@njit(cache=True)
def _segment_blocked(grid, ax, ay, bx, by, cell, jitter):
    """Same sampling as ``collision.segments_blocked``: every quarter cell, at least 10."""
    length = math.hypot(bx - ax, by - ay)
    count = max(int(length / (0.25 * cell)) + 1, 10)
    for i in range(count):
        t = i / (count - 1)
        if _blocked(grid, ax + t * (bx - ax), ay + t * (by - ay), cell, jitter):
            return True
    return False


@njit(cache=True)
def _energy(residual, cx, cy, radius, scale):
    bx = int(np.rint(cx))
    by = int(np.rint(cy))
    top = residual.shape[0] - 1
    total = 0.0
    for i in range(-radius, radius + 1):
        ix = min(max(bx + i + radius, 0), top)
        gx = math.exp(-((bx + i - cx) ** 2) / scale)
        row = 0.0
        for j in range(-radius, radius + 1):
            iy = min(max(by + j + radius, 0), top)
            row += residual[ix, iy] * math.exp(-((by + j - cy) ** 2) / scale)
        total += gx * row
    return total


@njit(cache=True)
def _consume(residual, cx, cy, radius, scale, amplitude):
    bx = int(np.rint(cx))
    by = int(np.rint(cy))
    top = residual.shape[0] - 1
    for i in range(-radius, radius + 1):
        ix = min(max(bx + i + radius, 0), top)
        gx = math.exp(-((bx + i - cx) ** 2) / scale)
        for j in range(-radius, radius + 1):
            iy = min(max(by + j + radius, 0), top)
            gy = math.exp(-((by + j - cy) ** 2) / scale)
            residual[ix, iy] = max(residual[ix, iy] - amplitude * gx * gy, 0.0)


@njit(parallel=True, cache=True)
def walk_batch(
    heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius, amplitude,
    arrival, max_steps, jitter,
):  # fmt: skip
    """``heat``/``grids`` ``(B, S, S)`` [x, y]; endpoints in pixels. Returns the pixel trail
    ``(B, max_steps + 2, 2)``, its lengths, statuses and the final residual (padded)."""
    b, s = heat.shape[0], heat.shape[1]
    scale = 2.0 * sigma * sigma
    trail = np.zeros((b, max_steps + 2, 2))
    lengths = np.ones(b, np.int64)
    status = np.full(b, ACTIVE, np.int64)
    residuals = np.zeros((b, s + 2 * radius, s + 2 * radius))
    angles = np.concatenate((forward, rear))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    n_forward, n = forward.shape[0], angles.shape[0]
    for q in prange(b):
        residual = residuals[q]
        residual[radius : radius + s, radius : radius + s] = heat[q]
        grid = grids[q]
        px, py = starts_px[q, 0], starts_px[q, 1]
        gx, gy = goals_px[q, 0], goals_px[q, 1]
        hx, hy = gx - px, gy - py
        norm = math.hypot(hx, hy)
        if norm > 0:
            hx, hy = hx / norm, hy / norm
        trail[q, 0, 0], trail[q, 0, 1] = px, py
        _consume(residual, px, py, radius, scale, amplitude)
        for _ in range(max_steps):
            if math.hypot(gx - px, gy - py) <= arrival and not _segment_blocked(
                grid, (px + 0.5) * cell, (py + 0.5) * cell, (gx + 0.5) * cell,
                (gy + 0.5) * cell, cell, jitter,
            ):  # fmt: skip
                px, py = gx, gy
                trail[q, lengths[q], 0], trail[q, lengths[q], 1] = px, py
                lengths[q] += 1
                status[q] = ARRIVED
                break
            best, best_e, bhx, bhy = -1, -np.inf, 0.0, 0.0
            for k in range(n):
                if k == n_forward and best >= 0:
                    break  # the rear sector only counts when every forward candidate is blocked
                dx = hx * cos_a[k] - hy * sin_a[k]
                dy = hy * cos_a[k] + hx * sin_a[k]
                cx, cy = px + step_px * dx, py + step_px * dy
                if _segment_blocked(
                    grid, (px + 0.5) * cell, (py + 0.5) * cell, (cx + 0.5) * cell,
                    (cy + 0.5) * cell, cell, jitter,
                ):  # fmt: skip
                    continue
                e = _energy(residual, cx, cy, radius, scale)
                if e > best_e:
                    best, best_e, bhx, bhy = k, e, dx, dy
            if best < 0:
                status[q] = STUCK
                break
            px, py = px + step_px * bhx, py + step_px * bhy
            hx, hy = bhx, bhy
            trail[q, lengths[q], 0], trail[q, lengths[q], 1] = px, py
            lengths[q] += 1
            _consume(residual, px, py, radius, scale, amplitude)
    return trail, lengths, status, residuals


@njit(parallel=True, cache=True)
def walk_batch_dfs(
    heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius, amplitude,
    arrival, max_steps, jitter, floor,
):  # fmt: skip
    """The walker with memory: a depth-first search over ridge branches.

    Every step it also notes the other peaks of the energy profile over its forward cone
    (energy >= ``floor``) as untried branches. When it is stuck, or the best energy ahead falls
    below ``floor`` (it has left the ridge), it returns to the latest branch, drops the steps
    it took since and goes that way. Walked ridge stays consumed, so a dead branch is not
    entered twice. With no branch left it falls back to reversing. ``max_steps`` bounds every
    step taken, explored or kept. Returns the kept trail, its length, statuses, residuals and
    the number of backtracks per query.
    """
    b, s = heat.shape[0], heat.shape[1]
    scale = 2.0 * sigma * sigma
    trail = np.zeros((b, max_steps + 2, 2))
    lengths = np.ones(b, np.int64)
    status = np.full(b, ACTIVE, np.int64)
    backtracks = np.zeros(b, np.int64)
    residuals = np.zeros((b, s + 2 * radius, s + 2 * radius))
    angles = np.concatenate((forward, rear))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    n_forward, n = forward.shape[0], angles.shape[0]
    cap = max_steps * 4 + 4
    for q in prange(b):
        residual = residuals[q]
        residual[radius : radius + s, radius : radius + s] = heat[q]
        grid = grids[q]
        # branch stack: kept-trail length at the branch, position, direction
        br_len = np.zeros(cap, np.int64)
        br_pos = np.zeros((cap, 2))
        br_dir = np.zeros((cap, 2))
        top = 0
        energy = np.empty(n)
        px, py = starts_px[q, 0], starts_px[q, 1]
        gx, gy = goals_px[q, 0], goals_px[q, 1]
        hx, hy = gx - px, gy - py
        norm = math.hypot(hx, hy)
        if norm > 0:
            hx, hy = hx / norm, hy / norm
        trail[q, 0, 0], trail[q, 0, 1] = px, py
        _consume(residual, px, py, radius, scale, amplitude)
        for _ in range(max_steps):
            if math.hypot(gx - px, gy - py) <= arrival and not _segment_blocked(
                grid, (px + 0.5) * cell, (py + 0.5) * cell, (gx + 0.5) * cell,
                (gy + 0.5) * cell, cell, jitter,
            ):  # fmt: skip
                px, py = gx, gy
                trail[q, lengths[q], 0], trail[q, lengths[q], 1] = px, py
                lengths[q] += 1
                status[q] = ARRIVED
                break
            best, best_e = -1, -np.inf
            for k in range(n_forward):
                dx = hx * cos_a[k] - hy * sin_a[k]
                dy = hy * cos_a[k] + hx * sin_a[k]
                cx, cy = px + step_px * dx, py + step_px * dy
                if _segment_blocked(
                    grid, (px + 0.5) * cell, (py + 0.5) * cell, (cx + 0.5) * cell,
                    (cy + 0.5) * cell, cell, jitter,
                ):  # fmt: skip
                    energy[k] = -np.inf
                    continue
                energy[k] = _energy(residual, cx, cy, radius, scale)
                if energy[k] > best_e:
                    best, best_e = k, energy[k]
            if best < 0 or best_e < floor:
                if top > 0:  # back to the latest untried branch
                    top -= 1
                    lengths[q] = br_len[top]
                    px, py = br_pos[top, 0], br_pos[top, 1]
                    hx, hy = br_dir[top, 0], br_dir[top, 1]
                    backtracks[q] += 1
                    cx, cy = px + step_px * hx, py + step_px * hy
                    px, py = cx, cy
                    trail[q, lengths[q], 0], trail[q, lengths[q], 1] = px, py
                    lengths[q] += 1
                    _consume(residual, px, py, radius, scale, amplitude)
                    continue
                if best < 0:  # nothing left to try ahead: the rear sector, as the greedy walker
                    best_e = -np.inf
                    for k in range(n_forward, n):
                        dx = hx * cos_a[k] - hy * sin_a[k]
                        dy = hy * cos_a[k] + hx * sin_a[k]
                        cx, cy = px + step_px * dx, py + step_px * dy
                        if _segment_blocked(
                            grid, (px + 0.5) * cell, (py + 0.5) * cell, (cx + 0.5) * cell,
                            (cy + 0.5) * cell, cell, jitter,
                        ):  # fmt: skip
                            continue
                        e = _energy(residual, cx, cy, radius, scale)
                        if e > best_e:
                            best, best_e = k, e
                    if best < 0:
                        status[q] = STUCK
                        break
            else:
                # remember the other peaks of the energy profile over the cone
                for k in range(n_forward):
                    if k == best or energy[k] < floor:
                        continue
                    left = energy[k - 1] if k > 0 else -np.inf
                    right = energy[k + 1] if k + 1 < n_forward else -np.inf
                    if energy[k] > left and energy[k] >= right and top < cap:
                        br_len[top] = lengths[q]
                        br_pos[top, 0], br_pos[top, 1] = px, py
                        br_dir[top, 0] = hx * cos_a[k] - hy * sin_a[k]
                        br_dir[top, 1] = hy * cos_a[k] + hx * sin_a[k]
                        top += 1
            dx = hx * cos_a[best] - hy * sin_a[best]
            dy = hy * cos_a[best] + hx * sin_a[best]
            px, py = px + step_px * dx, py + step_px * dy
            hx, hy = dx, dy
            trail[q, lengths[q], 0], trail[q, lengths[q], 1] = px, py
            lengths[q] += 1
            _consume(residual, px, py, radius, scale, amplitude)
    return trail, lengths, status, residuals, backtracks
