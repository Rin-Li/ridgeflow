from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ridgeflow.grid import GridSpec
from ridgeflow.model.targets import RIDGE_SIGMA, path_heatmap, world_to_pixel


class RidgeDataset(Dataset):
    """A pickled dict of ``map``, ``start``, ``goal``, ``paths`` (optional ``map_id``)."""

    def __init__(
        self,
        path: str | Path,
        grid: GridSpec | None = None,
        ridge_sigma: float = RIDGE_SIGMA,
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

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obstacle = torch.from_numpy(self.maps[self.map_ids[index]].T.copy()).unsqueeze(0)
        path_px = world_to_pixel(self.paths[index], self.grid)
        return {
            "obstacle": obstacle,
            "start_pixel": torch.from_numpy(world_to_pixel(self.starts[index], self.grid)),
            "goal_pixel": torch.from_numpy(world_to_pixel(self.goals[index], self.grid)),
            "target": path_heatmap(path_px, self.grid.size, self.ridge_sigma),
        }
