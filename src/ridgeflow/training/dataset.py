from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ridgeflow.grid import GridSpec
from ridgeflow.model.targets import RIDGE_SIGMA, geodesic_ridges, path_heatmap, world_to_pixel


class RidgeDataset(Dataset):
    """A pickled dict of ``map``, ``start``, ``goal``, ``paths`` (optional ``map_id``).

    ``geodesic``  measure the ridge's distance to the demonstration inside free space
                  (``targets.geodesic_ridges``), so no ridge leaks through a thin wall. The
                  targets are computed once, up front, on the GPU when there is one.
    ``augment``   apply one of the 8 symmetries of the square (rotations by 90 degrees and
                  mirror images), drawn uniformly per query, to the map, the endpoints and the
                  target alike. A path stays a valid path under all of them.
    """

    def __init__(
        self,
        path: str | Path,
        grid: GridSpec | None = None,
        ridge_sigma: float = RIDGE_SIGMA,
        geodesic: bool = False,
        augment: bool = False,
    ) -> None:
        data = np.load(Path(path), allow_pickle=True).item()
        self.grid = grid or GridSpec()
        self.ridge_sigma = float(ridge_sigma)
        self.maps = np.asarray(data["map"], np.float32)
        self.starts = np.asarray(data["start"], np.float32)
        self.goals = np.asarray(data["goal"], np.float32)
        self.paths = [np.asarray(p, np.float32) for p in data["paths"]]

        if len(self.maps) == len(self.starts):
            self.map_ids = np.arange(len(self.starts), dtype=np.int64)
        elif "map_id" in data:
            self.map_ids = np.asarray(data["map_id"], np.int64)
        else:
            raise ValueError("compact map storage requires one map_id per query")
        if len(self.map_ids) != len(self.starts):
            raise ValueError("map_id count does not match the number of queries")

        self.augment = augment
        self.targets = None
        if geodesic:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            out = []
            for begin in range(0, len(self.starts), 256):
                rows = range(begin, min(begin + 256, len(self.starts)))
                blocked = torch.from_numpy(
                    np.stack([self.maps[self.map_ids[i]].T > 0.5 for i in rows])
                ).to(device)
                paths = [world_to_pixel(self.paths[i], self.grid) for i in rows]
                out.append(geodesic_ridges(paths, blocked, self.ridge_sigma).half().cpu())
            self.targets = torch.cat(out)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obstacle = torch.from_numpy(self.maps[self.map_ids[index]].T.copy()).unsqueeze(0)
        if self.targets is not None:
            target = self.targets[index][None].float()
        else:
            path_px = world_to_pixel(self.paths[index], self.grid)
            target = path_heatmap(path_px, self.grid.size, self.ridge_sigma)
        item = {
            "obstacle": obstacle,
            "start_pixel": torch.from_numpy(world_to_pixel(self.starts[index], self.grid)),
            "goal_pixel": torch.from_numpy(world_to_pixel(self.goals[index], self.grid)),
            "target": target,
        }
        if self.augment:
            # torch's generator: the DataLoader seeds it per worker and per epoch
            item = dihedral(item, int(torch.randint(8, ())), self.grid.size)
        return item


def dihedral(item: dict, k: int, size: int) -> dict:
    """Symmetry ``k`` (0-7) of the square, applied alike to the ``(1, S, S)`` [y, x] images and
    to the [x, y] endpoint pixels (cell centres at integers, so a mirror maps p to S - 1 - p):
    bit 0 swaps the axes, bit 1 mirrors x, bit 2 mirrors y. ``k = 0`` is the identity."""
    images = {key: item[key] for key in ("obstacle", "target")}
    points = {key: item[key].clone() for key in ("start_pixel", "goal_pixel")}
    if k & 1:
        images = {key: v.transpose(-1, -2) for key, v in images.items()}
        points = {key: v.flip(-1) for key, v in points.items()}
    if k & 2:
        images = {key: v.flip(-1) for key, v in images.items()}
        for v in points.values():
            v[0] = size - 1 - v[0]
    if k & 4:
        images = {key: v.flip(-2) for key, v in images.items()}
        for v in points.values():
            v[1] = size - 1 - v[1]
    return {**{key: v.contiguous() for key, v in images.items()}, **points}
