from __future__ import annotations

import numpy as np
import pytest
import torch

from ridgeflow import RidgeModel, sample_rect_world, stack
from ridgeflow.model.flow import FlowMatching
from ridgeflow.model.unet import build_unet

TINY = {"base_channels": 8, "time_emb_dim": 16, "channel_mults": (1, 2)}


def _reference(flow, model, obstacle, start, goal, steps, eta, seed):
    """The sampler written as one loop that draws its noise as it goes."""
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, *obstacle.shape[-2:])
    for i in range(steps):
        t, t_next = 1.0 - i / steps, 1.0 - (i + 1) / steps
        v = model(x, obstacle, start, goal, torch.full((x.shape[0],), t * flow.time_scale))
        deterministic = x - v * (t - t_next)
        if eta > 0.0 and t_next > 0.0:
            stochastic = (1.0 - t_next) * (x - t * v) + t_next * torch.randn_like(x)
            x = (1.0 - eta) * deterministic + eta * stochastic
        else:
            x = deterministic
    return x.clamp(0.0, 1.0)


@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_noise_drawn_up_front_reproduces_the_sampler(eta) -> None:
    torch.manual_seed(0)
    model = build_unet(TINY).eval()
    flow = FlowMatching()
    obstacle = (torch.rand(3, 1, 16, 16) < 0.2).float()
    start, goal = torch.rand(2, 3, 1, 16, 16)
    with torch.no_grad():
        ref = _reference(flow, model, obstacle, start, goal, 4, eta, seed=7)
        out = flow.sample(model, obstacle, start, goal, steps=4, eta=eta, seed=7)
    assert torch.allclose(out, ref, atol=1e-6)


def test_ridge_model_samples_heatmaps(grid, tmp_path) -> None:
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save({"model_config": TINY, "model": build_unet(TINY).state_dict()}, path)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(2)]
    occ, _, starts, goals = stack(worlds, "cpu")
    model = RidgeModel(path, grid, device="cpu", fast=True)  # no CUDA: fast is ignored
    assert not model.fast
    a = model.sample(occ, starts, goals, seed=3)
    assert a.shape == (2, grid.size, grid.size) and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, model.sample(occ, starts, goals, seed=3))
