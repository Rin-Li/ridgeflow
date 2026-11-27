from __future__ import annotations

import numpy as np
import pytest
import torch

from ridgeflow import RidgeModel, RidgeWalker, WalkerConfig, sample_rect_world, stack
from ridgeflow.fast import FastRidgeModel, FastRidgeWalker
from ridgeflow.model.unet import build_unet
from ridgeflow.routes import project


def _queries(grid, n):
    """Rect worlds with their projected noise heatmaps: bright, broken and branching ridge."""
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(n)]
    _, grids, starts, goals = stack(worlds, "cpu")
    torch.manual_seed(0)
    h = torch.rand(n, grid.size, grid.size)
    si = torch.floor(starts / grid.cell).long()
    gi = torch.floor(goals / grid.cell).long()
    hp, route = project(h, ~grids, si, gi)
    return hp, grids, starts, goals, route


@pytest.mark.parametrize("backtrack", [True, False])
@pytest.mark.parametrize("batch", [1, 6])
def test_fast_walker_takes_the_same_steps(grid, backtrack, batch) -> None:
    hp, grids, starts, goals, route = _queries(grid, batch)
    config = WalkerConfig(backtrack=backtrack)
    a = RidgeWalker(grid, config).walk(hp, grids, starts, goals, route=route)
    b = FastRidgeWalker(grid, config).walk(hp, grids, starts, goals, route=route)
    assert np.array_equal(a.reached, b.reached)
    for p, q in zip(a.paths, b.paths, strict=True):
        assert np.array_equal(p, q)
    assert torch.equal(a.residual, b.residual)


def test_fast_model_in_fp32_reproduces_the_model(grid, tmp_path) -> None:
    config = {"base_channels": 8, "time_emb_dim": 16, "channel_mults": (1, 2)}
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save({"model_config": config, "model": build_unet(config).state_dict()}, path)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(3)]
    occ, _, starts, goals = stack(worlds, "cpu")
    ref = RidgeModel(path, grid, device="cpu").sample(occ, starts, goals, seed=7)
    fast = FastRidgeModel(path, grid, device="cpu", dtype=None, compile=False)
    assert torch.allclose(fast.sample(occ, starts, goals, seed=7), ref, atol=1e-6)

