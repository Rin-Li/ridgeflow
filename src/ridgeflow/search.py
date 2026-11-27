"""Grid search on 8-connected occupancy grids: A* and heatmap-guided FOCAL.

Moves go to the 8 neighbours of a free cell; a diagonal move needs both orthogonal neighbours
free (no corner cutting), the same rule as ``ridgeflow.routes``. Step costs are 1 and sqrt(2).
Every search counts the nodes it expands (a node is expanded once).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)
_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class SearchResult:
    path: np.ndarray | None  # (L, 2) cells, start first
    cost: float
    expanded: int
    closed: set | None = None  # expanded cells, when asked for


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def neighbours(blocked: np.ndarray, x: int, y: int):
    s0, s1 = blocked.shape
    for dx, dy in _MOVES:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < s0 and 0 <= ny < s1) or blocked[nx, ny]:
            continue
        if dx and dy and (blocked[x + dx, y] or blocked[x, y + dy]):
            continue
        yield nx, ny, (SQRT2 if dx and dy else 1.0)


def _path(parent: dict, node) -> np.ndarray:
    out = [node]
    while parent[out[-1]] is not None:
        out.append(parent[out[-1]])
    return np.array(out[::-1], np.int64)


def astar(blocked: np.ndarray, start, goal) -> SearchResult:
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    heap = [(octile(start, goal), 0.0, start)]
    expanded = 0
    while heap:
        f, gc, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            return SearchResult(_path(parent, node), gc, expanded)
        for nx, ny, c in neighbours(blocked, *node):
            nxt = (nx, ny)
            ng = gc + c
            if nxt not in closed and ng < g.get(nxt, math.inf):
                g[nxt] = ng
                parent[nxt] = node
                heapq.heappush(heap, (ng + octile(nxt, goal), ng, nxt))
    return SearchResult(None, math.inf, expanded)


def focal(
    blocked: np.ndarray, start, goal, guide: np.ndarray, w: float = 2.0, keep_closed: bool = False
) -> SearchResult:
    """FOCAL search: among open nodes with f <= w * f_min, expand the one with the largest
    ``guide`` value (ties: smaller f). ``guide`` only orders expansions; g, h and f come from
    the grid. With a consistent heuristic f_min never decreases, so a node that enters FOCAL
    stays inside the bound; nodes above it wait in ``pending`` until the bound reaches them."""
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    f0 = octile(start, goal)
    open_f = [(f0, start)]  # every open entry, for f_min
    pending: list = []  # open entries above the bound, by f
    focal_heap = [(-float(guide[start]), f0, start)]
    expanded = 0

    def stale(f, node):
        return node in closed or f > g[node] + octile(node, goal) + 1e-9

    while True:
        while open_f and stale(*open_f[0]):
            heapq.heappop(open_f)
        if not open_f:
            return SearchResult(None, math.inf, expanded)
        bound = w * open_f[0][0] + 1e-9
        while pending and pending[0][0] <= bound:
            f, node = heapq.heappop(pending)
            if not stale(f, node):
                heapq.heappush(focal_heap, (-float(guide[node]), f, node))
        node = None
        while focal_heap:
            _, f, cand = heapq.heappop(focal_heap)
            if not stale(f, cand):
                node = cand
                break
        if node is None:  # cannot happen while open_f is non-empty; guard anyway
            return SearchResult(None, math.inf, expanded)
        closed.add(node)
        expanded += 1
        if node == goal:
            return SearchResult(
                _path(parent, node), g[node], expanded, closed if keep_closed else None
            )
        for nx, ny, c in neighbours(blocked, *node):
            nxt = (nx, ny)
            ng = g[node] + c
            if nxt not in closed and ng < g.get(nxt, math.inf):
                g[nxt] = ng
                parent[nxt] = node
                f = ng + octile(nxt, goal)
                heapq.heappush(open_f, (f, nxt))
                if f <= bound:
                    heapq.heappush(focal_heap, (-float(guide[nxt]), f, nxt))
                else:
                    heapq.heappush(pending, (f, nxt))


def distances_from(blocked: np.ndarray, source) -> np.ndarray:
    """Exact octile distances from ``source`` to every cell (inf where unreachable)."""
    source = tuple(map(int, source))
    dist = np.full(blocked.shape, math.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for nx, ny, c in neighbours(blocked, *node):
            if d + c < dist[nx, ny]:
                dist[nx, ny] = d + c
                heapq.heappush(heap, (d + c, (nx, ny)))
    return dist
