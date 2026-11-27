from __future__ import annotations

import numpy as np
import pytest
import torch

from ridgeflow.model.targets import geodesic_ridges, path_heatmap
from ridgeflow.training.dataset import RidgeDataset, dihedral
from ridgeflow.training.gap import gap_loss
from ridgeflow.training.trainer import TrainConfig, train

S = 16


def _item(start=(2.0, 3.0), goal=(12.0, 9.0)) -> dict:
    """A straight demonstration on an empty map, with one obstacle cell to break symmetry."""
    obstacle = torch.zeros(1, S, S)
    obstacle[0, 10, 4] = 1.0  # [y, x]
    path = np.array([start, goal], np.float32)
    return {
        "obstacle": obstacle,
        "start_pixel": torch.tensor(start),
        "goal_pixel": torch.tensor(goal),
        "target": path_heatmap(path, S),
    }


@pytest.mark.parametrize("k", range(8))
def test_dihedral_moves_endpoints_with_the_images(k) -> None:
    item = _item()
    out = dihedral(item, k, S)
    for key in ("start_pixel", "goal_pixel"):  # the target ridge still peaks at both endpoints
        x, y = out[key].round().long().tolist()
        assert out["target"][0, y, x] == pytest.approx(1.0, abs=1e-5)
    ys, xs = torch.nonzero(out["obstacle"][0], as_tuple=True)
    assert len(ys) == 1  # one obstacle cell, moved like a point
    p = torch.tensor([4.0, 10.0])  # its [x, y]
    moved = dihedral({**item, "start_pixel": p, "goal_pixel": p}, k, S)["start_pixel"]
    assert moved.tolist() == [float(xs[0]), float(ys[0])]


def test_dihedral_gives_eight_distinct_symmetries() -> None:
    item = _item()
    seen = {tuple(dihedral(item, k, S)["target"].flatten().tolist()) for k in range(8)}
    assert len(seen) == 8
    assert torch.equal(dihedral(item, 0, S)["target"], item["target"])


def test_gap_loss_lifts_only_the_gap_on_the_route() -> None:
    item = _item(start=(2.0, 8.0), goal=(13.0, 8.0))
    h = item["target"].clone()[None].requires_grad_(True)  # (1, 1, S, S)
    obstacle = torch.zeros(1, 1, S, S)
    args = (obstacle, item["start_pixel"][None], item["goal_pixel"][None], torch.ones(1))
    loss, cells = gap_loss(h, *args)
    assert loss.item() == 0.0 and cells.item() == 0.0  # an unbroken ridge has no gap

    broken = item["target"].clone()
    broken[0, :, 7:9] = 0.0  # cut the ridge two cells wide
    h = broken[None].requires_grad_(True)
    loss, cells = gap_loss(h, *args)
    assert cells.item() >= 2 and loss.item() > 0.0
    loss.backward()
    lifted = h.grad[0, 0] < 0  # descending the loss raises h exactly there
    assert lifted.any() and lifted[:, 7:9].sum() == lifted.sum()


def test_gap_loss_is_weighted_per_sample() -> None:
    item = _item(start=(2.0, 8.0), goal=(13.0, 8.0))
    broken = item["target"].clone()
    broken[0, :, 7:9] = 0.0
    h = broken[None].repeat(2, 1, 1, 1)
    args = (
        torch.zeros(2, 1, S, S),
        item["start_pixel"].repeat(2, 1),
        item["goal_pixel"].repeat(2, 1),
    )
    full, _ = gap_loss(h, *args, torch.tensor([1.0, 1.0]))
    half, _ = gap_loss(h, *args, torch.tensor([1.0, 0.0]))
    assert float(half) == pytest.approx(float(full) / 2)


def test_geodesic_ridge_does_not_leak_through_a_wall() -> None:
    blocked = torch.zeros(1, S, S, dtype=torch.bool)
    blocked[0, 8, 1:15] = True  # a one-cell wall under the path, [y, x]
    path = np.array([[1.0, 6.0], [14.0, 6.0]], np.float32)  # [x, y], two cells above the wall
    geo = geodesic_ridges([path], blocked)[0]
    euclid = path_heatmap(path, S)[0]
    assert geo[8, 1:15].abs().max() == 0.0  # obstacle cells are 0
    assert geo[10, 7] < 0.5 * euclid[10, 7]  # behind the wall: far around, not straight through
    # on the path's side: the same ridge, up to the octile approximation of distance
    assert torch.allclose(geo[5:8], euclid[5:8], atol=1e-5)
    assert torch.allclose(geo[:8], euclid[:8], atol=0.1)


def test_training_with_gap_loss_and_augmentation_runs(tmp_path) -> None:
    rng = np.random.default_rng(0)
    maps = np.zeros((4, S, S), np.float32)
    maps[:, 6:9, 6:9] = 1.0
    starts = rng.uniform(0.1, 0.5, size=(4, 2)).astype(np.float32)
    goals = rng.uniform(1.5, 1.9, size=(4, 2)).astype(np.float32)
    paths = [np.stack([s, [0.3, 1.8], g]) for s, g in zip(starts, goals, strict=True)]
    data = {"map": maps, "start": starts, "goal": goals, "paths": paths}
    np.save(tmp_path / "data.npy", data, allow_pickle=True)

    from ridgeflow import GridSpec

    grid = GridSpec(size=S, extent=2.0)
    dataset = RidgeDataset(tmp_path / "data.npy", grid, geodesic=True, augment=True)
    config = TrainConfig(
        output_dir=tmp_path / "run", epochs=1, batch_size=2, grad_accum_steps=1, val_ratio=0.0,
        val_every=1, loader_workers=0, amp=False, gap_weight=100.0,
    )  # fmt: skip
    config.model.update(base_channels=8, time_emb_dim=16, channel_mults=(1, 2))
    train(dataset, config, dataset, grid, device="cpu")
    assert (tmp_path / "run" / "best.pt").exists()
