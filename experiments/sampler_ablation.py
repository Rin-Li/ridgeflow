"""Sampler ablation, no retraining: where to project, a rising threshold, and self-recurrence.

Every variant is the shipped 4-step sampler plus, per step i (t_i = 1 - i/4):
  * projection (if step i is in ``project_at``): when b*(x0_hat) < theta_i, replace x0_hat by
    its L2 projection onto {H : some free route has H >= theta_i}, i.e. lift the cells of the
    cheapest route under (theta_i - H)_+^2 to theta_i;
  * self-recurrence (Universal Guidance): run the step ``recur`` times; between repeats the new
    state is re-noised back to t_i with the rectified-flow forward kernel
    x_t = a x_t' + sqrt(t^2 - a^2 t'^2) xi,  a = (1 - t) / (1 - t').
Metrics: output connectivity, hedging (second unbroken route), walker / geodesic success,
diversity across seeds, network evaluations and time per map. Writes a results PDF.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))
from compare_report import draw  # noqa: E402
from connectivity_eval import diversity  # noqa: E402
from topology_report import base_axes  # noqa: E402  (registers the CJK font)

from ridgeflow import GridSpec, RidgeModel, RidgeWalker, sample_rect_world, stack  # noqa: E402
from ridgeflow.routes import (  # noqa: E402
    ON_RIDGE,
    bottleneck,
    hedging,
    projection_cost,
    ridge_cost,
    route_mask,
    route_paths,
)


@dataclass(frozen=True)
class Variant:
    name: str
    project_at: tuple[int, ...] = ()
    thetas: tuple[float, ...] = (ON_RIDGE,) * 4
    recur: int = 1
    note: str = ""


ALL = (0, 1, 2, 3)
RAMP = (0.2, 0.3, 0.4, 0.5)
VARIANTS = [
    Variant("plain", note="no guidance"),
    Variant("last", (3,), note="project at last step only"),
    Variant("last2", (2, 3), note="project at last two steps"),
    Variant("all", ALL, note="project every step (θ=0.5)"),
    Variant("ramp", ALL, RAMP, note="project every step, θ: 0.2→0.5"),
    Variant("plain_r2", recur=2, note="no guidance + 2 repeats per step"),
    Variant("all_r2", ALL, recur=2, note="project every step + 2 repeats"),
    Variant("ramp_r2", ALL, RAMP, recur=2, note="rising θ + 2 repeats"),
]


@torch.no_grad()
def sample(model: RidgeModel, occ, grids, starts, goals, seed: int, v: Variant, trace=False):
    flow, net, dev = model.flow, model.model, model.device
    cell = model.grid.cell
    obstacle = occ.to(dev, torch.float32).transpose(1, 2)[:, None]
    start_h, goal_h = model._endpoints(starts), model._endpoints(goals)
    si = torch.floor(starts / cell).long()
    gi = torch.floor(goals / cell).long()
    free = ~grids
    torch.manual_seed(seed)
    x = torch.randn(obstacle.shape[0], 1, *obstacle.shape[-2:], device=dev)
    steps, eta = model.steps, model.eta
    schedule = [1.0 - i / steps for i in range(steps + 1)]
    frames: list = []
    projected = [0] * steps
    for i in range(steps):
        t, t_next = schedule[i], schedule[i + 1]
        tt = torch.full((x.shape[0],), t * flow.time_scale, device=dev)
        for r in range(v.recur):
            vel = net(x, obstacle, start_h, goal_h, tt)
            x0 = x - t * vel
            h = x0[:, 0].transpose(1, 2)  # [x, y]
            before = h.clone()
            if i in v.project_at:
                theta = v.thetas[i]
                broken = bottleneck(h, free, si, gi) < theta
                if bool(broken.any()):
                    rows = broken.nonzero()[:, 0]
                    lift = route_mask(
                        projection_cost(h[rows], theta), free[rows], si[rows], gi[rows]
                    )
                    h[rows] = torch.where(lift, h[rows].clamp_min(theta), h[rows])
                if r == v.recur - 1:
                    projected[i] = int(broken.sum())
            if trace and r == v.recur - 1:
                frames.append((before.cpu().numpy(), (h - before).abs().cpu().numpy() > 1e-6))
            x0 = h.transpose(1, 2)[:, None]
            if t_next > 0.0:
                noise = torch.randn_like(x)
                det = (1.0 - t_next) * x0 + t_next * (x - (1.0 - t) * x0) / t
                nxt = (1.0 - eta) * det + eta * ((1.0 - t_next) * x0 + t_next * noise)
            else:
                nxt = x0
            if r < v.recur - 1:  # re-noise back to t and redo the step
                a = (1.0 - t) / (1.0 - t_next)
                std = max(t**2 - (a * t_next) ** 2, 0.0) ** 0.5
                x = a * nxt + std * torch.randn_like(x)
            else:
                x = nxt
    return x.clamp(0, 1)[:, 0].transpose(1, 2).contiguous(), projected, frames


@dataclass
class Stats:
    conn: list = field(default_factory=list)
    hedge: list = field(default_factory=list)
    walker: list = field(default_factory=list)
    geo: list = field(default_factory=list)
    div: list = field(default_factory=list)
    projected: np.ndarray = field(default_factory=lambda: np.zeros(4))
    seconds: float = 0.0


def evaluate(model, worlds, walker, v: Variant, k: int, batch: int, keep: dict, mname: str):
    grid = model.grid
    st = Stats()
    for begin in range(0, len(worlds), batch):
        chunk = worlds[begin : begin + batch]
        occ, grids, starts, goals = stack(chunk, model.device)
        si = torch.floor(starts / grid.cell).long()
        gi = torch.floor(goals / grid.cell).long()
        masks = []
        for j in range(k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h, projected, _ = sample(model, occ, grids, starts, goals, begin + 10_000 * j, v)
            torch.cuda.synchronize()
            st.seconds += time.perf_counter() - t0
            st.projected += np.array(projected)
            st.conn += (bottleneck(h, ~grids, si, gi) >= ON_RIDGE).tolist()
            st.hedge += (hedging(h, ~grids, si, gi)[0] >= ON_RIDGE).tolist()
            res = walker.walk(h, grids, starts, goals)
            st.walker += [
                bool(r) and w.path_free(p)
                for w, p, r in zip(chunk, res.paths, res.reached, strict=True)
            ]
            cells = route_paths(ridge_cost(h), ~grids, si, gi)
            for w, c, s_, g_ in zip(chunk, cells, starts.cpu().numpy(), goals.cpu().numpy(),
                                    strict=True):
                if c is None:
                    st.geo.append(False)
                else:
                    pts = np.concatenate([s_[None], (c[1:-1] + 0.5) * grid.cell, g_[None]])
                    st.geo.append(w.path_free(pts))
            masks.append((h >= ON_RIDGE).cpu().numpy())
            if j == 0:
                hn = h.cpu().numpy()
                for q in range(len(chunk)):
                    keep[(mname, v.name, begin + q)] = hn[q]
        st.div += [diversity(m) for m in np.stack(masks, 1)]
    return st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="name=path")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output", default="outputs/sampler_ablation.pdf")
    args = parser.parse_args()

    grid = GridSpec()
    walker = RidgeWalker(grid)
    worlds = [sample_rect_world(np.random.default_rng(s), grid) for s in range(args.queries)]
    rows, keep, models = [], {}, {}
    for item in args.checkpoints:
        mname, path = item.split("=", 1)
        model = models[mname] = RidgeModel(path, grid)
        occ, grids, starts, goals = stack(worlds[:8], model.device)
        sample(model, occ, grids, starts, goals, 0, VARIANTS[3])  # warm-up
        for v in VARIANTS:
            st = evaluate(model, worlds, walker, v, args.k, args.batch, keep, mname)
            n = len(worlds) * args.k
            row = dict(
                model=mname, variant=v.name, note=v.note, nfe=4 * v.recur,
                conn=100 * np.mean(st.conn), hedge=100 * np.mean(st.hedge),
                walker=100 * np.mean(st.walker), geo=100 * np.mean(st.geo),
                div=float(np.mean(st.div)), ms=1000 * st.seconds / n,
                proj=" ".join(f"{100 * p / n:.0f}" for p in st.projected) if v.project_at else "—",
            )
            rows.append(row)
            print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "Sampler ablation: projection timing × threshold × self-recurrence",
                 ha="center", fontsize=15, weight="bold")
        intro = [
            f"{len(worlds)} benchmark maps × {args.k} noise seeds."
            "Projection = if b*(x̂0) < θ, raise the cells on the shortest route under cost "
            "(θ−H)₊² to θ (minimal L2 change).",
            "Heatmap connectivity = output H has a route from start to goal with H ≥ 0.5 "
            "throughout; multi-route = after removing the main route's 3px corridor, "
            "another route with H ≥ 0.5 throughout remains (hedging).",
            "Per-step projection % = share of samples judged disconnected and projected at steps "
            "1-4. NFE = network calls.",
        ]
        for i, s in enumerate(intro):
            fig.text(0.04, 0.89 - 0.03 * i, s, fontsize=9.5)
        cols = ["model", "variant", "note", "NFE", "heatmap conn.", "multi-route", "walker",
                "geodesic",
                "diversity", "ms/map", "per-step proj. %"]
        cells = [[r["model"], r["variant"], r["note"], r["nfe"], f"{r['conn']:.1f}%",
                  f"{r['hedge']:.1f}%", f"{r['walker']:.1f}%", f"{r['geo']:.1f}%",
                  f"{r['div']:.3f}", f"{r['ms']:.1f}", r["proj"]] for r in rows]
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.72])
        ax.axis("off")
        tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1, 1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # maps where plain is broken, for the first model
        first = args.checkpoints[0].split("=", 1)[0]
        model = models[first]
        shown_variants = ["plain", "last", "all", "ramp", "all_r2"]
        broken = []
        for s in range(len(worlds)):
            occ, grids, starts, goals = stack([worlds[s]], model.device)
            si = torch.floor(starts / grid.cell).long()
            gi = torch.floor(goals / grid.cell).long()
            h = torch.from_numpy(keep[(first, "plain", s)])[None].to(model.device)
            if float(bottleneck(h, ~grids, si, gi)) < ON_RIDGE:
                broken.append(s)
            if len(broken) == 12:
                break
        for p in range(0, len(broken), 4):
            fig, axes = plt.subplots(4, len(shown_variants), figsize=(11.69, 9.6))
            fig.suptitle(f"{first}: maps where plain breaks (blue = H>=0.5 on geodesic, red = gap)",
                         fontsize=11, weight="bold")
            for r, s in enumerate(broken[p : p + 4]):
                occ, grids, starts, goals = stack([worlds[s]], model.device)
                si = torch.floor(starts / grid.cell).long()
                gi = torch.floor(goals / grid.cell).long()
                for c, vn in enumerate(shown_variants):
                    h = torch.from_numpy(keep[(first, vn, s)])[None].to(model.device)
                    route = route_mask(ridge_cost(h), ~grids, si, gi)[0].cpu().numpy()
                    b = float(bottleneck(h, ~grids, si, gi))
                    b2 = float(hedging(h, ~grids, si, gi)[0])
                    draw(axes[r, c], worlds[s], h[0].cpu().numpy(), route, b,
                         f"#{s} {vn}  b*₂={b2:.2f}")
            for ax in axes[len(broken[p : p + 4]) :].ravel():
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)

        # per-step x0_hat for two broken maps: plain vs all vs ramp_r2
        trace_variants = [v for v in VARIANTS if v.name in ("plain", "all", "ramp_r2")]
        for s in broken[:2]:
            world = worlds[s]
            occ, grids, starts, goals = stack([world], model.device)
            fig, axes = plt.subplots(3, 5, figsize=(11.69, 7.4))
            fig.suptitle(f"map #{s}: x̂0 per step (cyan = cells raised by projection)",
                         fontsize=11, weight="bold")
            seed = (s // args.batch) * args.batch
            for r, v in enumerate(trace_variants):
                h, _, frames = sample(model, occ, grids, starts, goals, seed, v, trace=True)
                for c, (x0, lifted) in enumerate(frames):
                    base_axes(axes[r, c], world, np.clip(x0[0], 0, 1))
                    xs, ys = np.nonzero(lifted[0])
                    axes[r, c].scatter((xs + 0.5) * grid.cell, (ys + 0.5) * grid.cell, s=4,
                                       c="cyan", marker="s", lw=0, zorder=6)
                    axes[r, c].set_title(f"{v.name} step {c + 1} x̂0", fontsize=8)
                base_axes(axes[r, 4], world, h[0].cpu().numpy())
                axes[r, 4].set_title(f"{v.name} output", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
