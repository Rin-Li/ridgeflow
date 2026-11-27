"""ScaleCrafter-style re-dilation on TMP64, training-free: at the noisier sampling steps, run some
of the trained 3x3 convolutions with dilation d (no new parameters), then measure connectivity
and FOCAL expansions.

Layers: "mid" = the two bottleneck ResBlocks (4 convs, 16x16); "mid+deep" also the deepest
encoder and first decoder block (16x16). ``--until`` is the noise level t above which a step
is re-dilated (t = 1 is pure noise; with 4 steps the steps start at t = 1, .75, .5, .25).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tmp64_eval import _search  # noqa: E402

from ridgeflow import GridSpec, RidgeModel  # noqa: E402
from ridgeflow.routes import ON_RIDGE, bottleneck  # noqa: E402


def convs(net, where: str):
    blocks = [net.mid1, net.mid2]
    if where == "mid+deep":
        blocks += [net.downs[-1].res1, net.downs[-1].res2, net.ups[0].res1, net.ups[0].res2]
    return [c for b in blocks for c in (b.conv1, b.conv2)]


@contextmanager
def dilated(layers, d: int):
    old = [(c.dilation, c.padding) for c in layers]
    for c in layers:
        c.dilation, c.padding = (d, d), (d, d)
    try:
        yield
    finally:
        for c, (dil, pad) in zip(layers, old, strict=True):
            c.dilation, c.padding = dil, pad


@torch.no_grad()
def sample(model, occ, starts, goals, seed, layers, d, above):
    flow, net, dev = model.flow, model.model, model.device
    obstacle = occ.float().transpose(1, 2)[:, None]
    sh, gh = model._endpoints(starts), model._endpoints(goals)
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, 64, 64, device=dev)
    k, eta = model.steps, model.eta
    ts = [1.0 - i / k for i in range(k + 1)]
    for i in range(k):
        t, tn = ts[i], ts[i + 1]
        tt = torch.full((x.shape[0],), t * flow.time_scale, device=dev)
        if layers and t > above:
            with dilated(layers, d):
                v = net(x, obstacle, sh, gh, tt)
        else:
            v = net(x, obstacle, sh, gh, tt)
        det = x - v * (t - tn)
        if eta > 0 and tn > 0:
            x0 = x - t * v
            x = (1 - eta) * det + eta * ((1 - tn) * x0 + tn * torch.randn_like(x))
        else:
            x = det
    return x.clamp(0, 1)[:, 0].transpose(1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/runs/tmp_s15_geo_gap100/best.pt")
    parser.add_argument("--data", default="outputs/data/tmp64_test.npy")
    args = parser.parse_args()
    data = np.load(args.data, allow_pickle=True).item()
    n = len(data["start"])
    blocked = [data["map"][data["map_id"][i]].astype(bool) for i in range(n)]
    sc = [tuple(c) for c in data["start_cell"]]
    gc = [tuple(c) for c in data["goal_cell"]]
    pool = Pool(20)
    base = pool.map(_search, [(blocked[i], sc[i], gc[i], None, 1.0) for i in range(n)])
    a_exp = np.array([e for e, _ in base], float)
    a_cost = np.array([c for _, c in base])
    model = RidgeModel(args.checkpoint, GridSpec())
    configs = [
        ("none", None, 1, 1.0),
        ("mid d2 early", "mid", 2, 0.6),
        ("mid d2 all", "mid", 2, -1.0),
        ("mid d3 early", "mid", 3, 0.6),
        ("mid+deep d2 early", "mid+deep", 2, 0.6),
        ("mid d2 first", "mid", 2, 0.9),
    ]
    print(f"{'config':<20}{'connected':>10}{'walls':>8}{'exp.red':>9}{'cost':>7}")
    for name, where, d, above in configs:
        layers = convs(model.model, where) if where else []
        conn, cross, red, cost = [], [], [], []
        for begin in range(0, n, 128):
            idx = np.arange(begin, min(begin + 128, n))
            occ = torch.from_numpy(data["map"][data["map_id"][idx]]).bool().cuda()
            st = torch.from_numpy(data["start"][idx]).cuda()
            go = torch.from_numpy(data["goal"][idx]).cuda()
            si = torch.from_numpy(data["start_cell"][idx]).cuda()
            gi = torch.from_numpy(data["goal_cell"][idx]).cuda()
            h = sample(model, occ, st, go, begin, layers, d, above)
            b = bottleneck(h, ~occ, si, gi)
            w = bottleneck(h, torch.ones_like(occ), si, gi)
            conn += (b >= ON_RIDGE).tolist()
            cross += ((b < ON_RIDGE) & (w >= ON_RIDGE)).tolist()
            hn = h.cpu().numpy()
            res = pool.map(
                _search, [(blocked[i], sc[i], gc[i], hn[q], 2.0) for q, i in enumerate(idx)]
            )
            red += [100 * (1 - r[0] / a_exp[i]) for r, i in zip(res, idx, strict=True)]
            cost += [r[1] / a_cost[i] for r, i in zip(res, idx, strict=True)]
        print(
            f"{name:<20}{100 * np.mean(conn):>9.1f}%{100 * np.mean(cross):>7.1f}%"
            f"{np.mean(red):>8.1f}%{np.mean(cost):>7.3f}",
            flush=True,
        )
    pool.close()


if __name__ == "__main__":
    main()
