from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Result:
    """World-frame paths for a batch of queries, and which of them reached the goal."""

    paths: list[np.ndarray]
    reached: np.ndarray
    residual: torch.Tensor | None = None


def collect(trail: list[torch.Tensor], lengths: torch.Tensor) -> list[np.ndarray]:
    """``trail`` holds one ``(B, 2)`` position per step; keep each query's first ``lengths``."""
    stacked = torch.stack(trail, dim=1).cpu().numpy()
    return [stacked[b, :n] for b, n in enumerate(lengths.tolist())]
