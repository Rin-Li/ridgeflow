"""TMP64: does more sampling steps (more passes of the same network) connect more heatmaps?"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np

from experiments.common import (
    PLAIN,
    TMP64_CHECKPOINT,
    TMP64_DATA,
    astar_baseline,
    search,
    tmp64_heatmaps,
    tmp64_queries,
)
from ridgeflow import GridSpec, RidgeModel
from ridgeflow.routes import ON_RIDGE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=TMP64_CHECKPOINT)
    parser.add_argument("--data", default=TMP64_DATA)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--eta", type=float, nargs="+", default=[0.5])
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True).item()
    blocked, sc, gc = tmp64_queries(data)
    n = len(blocked)
    pool = Pool(20)
    a_exp, a_cost = astar_baseline(pool, blocked, sc, gc)
    model = RidgeModel(args.checkpoint, GridSpec())
    print(
        f"{'K':>3} {'eta':>4} {'connected':>10} {'through walls':>14} {'exp. red.':>10} {'cost':>7}"
    )
    for eta in args.eta:
        for k in args.steps:
            model.steps, model.eta = k, eta
            conn, cross, red, cost = [], [], [], []
            for begin in range(0, n, 128):
                idx = np.arange(begin, min(begin + 128, n))
                h, b, w = tmp64_heatmaps(model, data, idx, begin, PLAIN, 1)
                w = w[:, 0]
                b = b[:, 0]
                conn += (b >= ON_RIDGE).tolist()
                cross += ((b < ON_RIDGE) & (w >= ON_RIDGE)).tolist()
                hn = h[:, 0].cpu().numpy()
                res = pool.map(
                    search, [(blocked[i], sc[i], gc[i], hn[q], 2.0) for q, i in enumerate(idx)]
                )
                red += [100 * (1 - r[0] / a_exp[i]) for r, i in zip(res, idx, strict=True)]
                cost += [r[1] / a_cost[i] for r, i in zip(res, idx, strict=True)]
            print(
                f"{k:>3} {eta:>4} {100 * np.mean(conn):>9.1f}% {100 * np.mean(cross):>13.1f}% "
                f"{np.mean(red):>9.1f}% {np.mean(cost):>7.3f}",
                flush=True,
            )
    pool.close()


if __name__ == "__main__":
    main()
