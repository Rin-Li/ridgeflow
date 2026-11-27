"""Faster numba kernels for the ridge walker: the same walks as ``_walker_numba``, bit for bit.

Two changes, neither of which alters a single result:
  * the Gaussian window weights are computed once per axis (2 x 15 ``exp``) instead of once per
    window cell (225), in both the energy read and the consumption;
  * a segment's collision check first looks at the cells of its bounding box widened by the
    largest jitter probe; when they are all on the grid and free, no probe can be blocked, so
    the ~130 probe lookups are skipped. Otherwise it falls back to the probes.
Each query's walk is its own function, so the batch runs either on threads (``prange``) or on
the calling thread; for one or two queries the thread pool costs more than the walks.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ridgeflow.planners._walker_numba import _JITTER, ACTIVE, ARRIVED, STUCK, _blocked

__all__ = ["ACTIVE", "ARRIVED", "STUCK", "_JITTER", "walk_batch", "walk_batch_dfs"]

SERIAL_UP_TO = 2  # batches this small run on the calling thread


@njit(cache=True)
def _segment_blocked(grid, ax, ay, bx, by, cell, jitter):
    s = grid.shape[0]
    j = 0.0
    for k in range(jitter.shape[0]):
        j = max(j, abs(jitter[k, 0]), abs(jitter[k, 1]))
    x0 = int(math.floor(min(ax, bx) / cell - j - 1e-9))
    x1 = int(math.floor(max(ax, bx) / cell + j + 1e-9))
    y0 = int(math.floor(min(ay, by) / cell - j - 1e-9))
    y1 = int(math.floor(max(ay, by) / cell + j + 1e-9))
    if x0 >= 0 and y0 >= 0 and x1 < s and y1 < s and (x1 - x0 + 1) * (y1 - y0 + 1) <= 16:
        clear = True
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                if grid[ix, iy]:
                    clear = False
        if clear:
            return False
    # the probes of ``_walker_numba._segment_blocked``: every quarter cell, at least 10
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
    w = 2 * radius + 1
    gys = np.empty(w)
    iys = np.empty(w, np.int64)
    for j in range(w):
        gys[j] = math.exp(-((by + j - radius - cy) ** 2) / scale)
        iys[j] = min(max(by + j, 0), top)
    total = 0.0
    for i in range(-radius, radius + 1):
        ix = min(max(bx + i + radius, 0), top)
        gx = math.exp(-((bx + i - cx) ** 2) / scale)
        row = 0.0
        for j in range(w):
            row += residual[ix, iys[j]] * gys[j]
        total += gx * row
    return total


@njit(cache=True)
def _consume(residual, cx, cy, radius, scale, amplitude):
    bx = int(np.rint(cx))
    by = int(np.rint(cy))
    top = residual.shape[0] - 1
    w = 2 * radius + 1
    gys = np.empty(w)
    iys = np.empty(w, np.int64)
    for j in range(w):
        gys[j] = math.exp(-((by + j - radius - cy) ** 2) / scale)
        iys[j] = min(max(by + j, 0), top)
    for i in range(-radius, radius + 1):
        ix = min(max(bx + i + radius, 0), top)
        gx = math.exp(-((bx + i - cx) ** 2) / scale)
        for j in range(w):
            residual[ix, iys[j]] = max(residual[ix, iys[j]] - amplitude * gx * gys[j], 0.0)


@njit(cache=True)
def _walk_one(
    q, heat, grids, starts_px, goals_px, cell, step_px, cos_a, sin_a, n_forward, radius, scale,
    amplitude, arrival, max_steps, jitter, trail, lengths, status, residuals,
):  # fmt: skip
    """One greedy walk (``_walker_numba.walk_batch``'s loop body)."""
    s = heat.shape[1]
    n = cos_a.shape[0]
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
                break
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


@njit(cache=True)
def _walk_one_dfs(
    q, heat, grids, starts_px, goals_px, cell, step_px, cos_a, sin_a, n_forward, radius, scale,
    amplitude, arrival, max_steps, jitter, floor, trail, lengths, status, residuals, backtracks,
):  # fmt: skip
    """One backtracking walk (``_walker_numba.walk_batch_dfs``'s loop body)."""
    s = heat.shape[1]
    n = cos_a.shape[0]
    cap = max_steps * 4 + 4
    residual = residuals[q]
    residual[radius : radius + s, radius : radius + s] = heat[q]
    grid = grids[q]
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
            if top > 0:
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
            if best < 0:
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


def _buffers(heat, radius, max_steps):
    b, s = heat.shape[0], heat.shape[1]
    return (
        np.zeros((b, max_steps + 2, 2)),
        np.ones(b, np.int64),
        np.full(b, ACTIVE, np.int64),
        np.zeros((b, s + 2 * radius, s + 2 * radius)),
    )


@njit(parallel=True, cache=True)
def _greedy_threads(args_a, cos_a, sin_a, n_forward, args_b, out):
    trail, lengths, status, residuals = out
    for q in prange(args_a[0].shape[0]):
        _walk_one(q, *args_a, cos_a, sin_a, n_forward, *args_b, trail, lengths, status, residuals)


@njit(cache=True)
def _greedy_serial(args_a, cos_a, sin_a, n_forward, args_b, out):
    trail, lengths, status, residuals = out
    for q in range(args_a[0].shape[0]):
        _walk_one(q, *args_a, cos_a, sin_a, n_forward, *args_b, trail, lengths, status, residuals)


@njit(parallel=True, cache=True)
def _dfs_threads(args_a, cos_a, sin_a, n_forward, args_b, out, backtracks):
    trail, lengths, status, residuals = out
    for q in prange(args_a[0].shape[0]):
        _walk_one_dfs(
            q, *args_a, cos_a, sin_a, n_forward, *args_b, trail, lengths, status, residuals,
            backtracks,
        )  # fmt: skip


@njit(cache=True)
def _dfs_serial(args_a, cos_a, sin_a, n_forward, args_b, out, backtracks):
    trail, lengths, status, residuals = out
    for q in range(args_a[0].shape[0]):
        _walk_one_dfs(
            q, *args_a, cos_a, sin_a, n_forward, *args_b, trail, lengths, status, residuals,
            backtracks,
        )  # fmt: skip


def _split(heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius,
           amplitude, arrival, max_steps, jitter):  # fmt: skip
    angles = np.concatenate((forward, rear))
    args_a = (heat, grids, starts_px, goals_px, float(cell), float(step_px))
    args_b = (int(radius), 2.0 * sigma * sigma, float(amplitude), float(arrival), int(max_steps),
              jitter)  # fmt: skip
    return args_a, np.cos(angles), np.sin(angles), int(forward.shape[0]), args_b


def walk_batch(heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius,
               amplitude, arrival, max_steps, jitter):  # fmt: skip
    """Same arguments and returns as ``_walker_numba.walk_batch``."""
    args_a, cos_a, sin_a, n_forward, args_b = _split(
        heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius, amplitude,
        arrival, max_steps, jitter,
    )  # fmt: skip
    out = _buffers(heat, radius, max_steps)
    run = _greedy_serial if heat.shape[0] <= SERIAL_UP_TO else _greedy_threads
    run(args_a, cos_a, sin_a, n_forward, args_b, out)
    return out


def walk_batch_dfs(heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius,
                   amplitude, arrival, max_steps, jitter, floor):  # fmt: skip
    """Same arguments and returns as ``_walker_numba.walk_batch_dfs``."""
    args_a, cos_a, sin_a, n_forward, args_b = _split(
        heat, grids, starts_px, goals_px, cell, step_px, forward, rear, sigma, radius, amplitude,
        arrival, max_steps, jitter,
    )  # fmt: skip
    args_b = (*args_b, float(floor))
    out = _buffers(heat, radius, max_steps)
    backtracks = np.zeros(heat.shape[0], np.int64)
    run = _dfs_serial if heat.shape[0] <= SERIAL_UP_TO else _dfs_threads
    run(args_a, cos_a, sin_a, n_forward, args_b, out, backtracks)
    return (*out, backtracks)

