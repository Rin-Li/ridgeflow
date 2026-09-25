from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from ridgeflow.grid import GridSpec
from ridgeflow.model.flow import FlowMatching
from ridgeflow.model.targets import ENDPOINT_SIGMA, endpoint_heatmaps
from ridgeflow.model.unet import build_unet


@dataclass
class TrainConfig:
    """The defaults reproduce the shipped checkpoint."""

    output_dir: Path = Path("checkpoints/run")
    epochs: int = 300
    batch_size: int = 16
    grad_accum_steps: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 100
    ema_decay: float = 0.999
    val_ratio: float = 0.05
    val_every: int = 2
    early_stop_patience: int = 25
    loader_workers: int = 4
    amp: bool = True
    seed: int = 0
    endpoint_sigma: float = ENDPOINT_SIGMA
    time_scale: float = 1000.0
    sigma_min: float = 0.0
    t_mode: str = "uniform"
    model: dict = field(
        default_factory=lambda: {
            "in_channels": 4,
            "out_channels": 1,
            "base_channels": 64,
            "time_emb_dim": 256,
            "channel_mults": (1, 2, 4),
        }
    )


def _warmup_cosine(total: int, warmup: int):
    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def _split(dataset: Dataset, val_ratio: float, seed: int):
    if val_ratio <= 0.0:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_ratio))
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [len(dataset) - val_size, val_size], generator=generator)


def train(
    dataset: Dataset,
    config: TrainConfig | None = None,
    val_dataset: Dataset | None = None,
    grid: GridSpec | None = None,
    device: str = "cuda",
) -> Path:
    config = config or TrainConfig()
    grid = grid or GridSpec()
    torch.manual_seed(config.seed)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    if val_dataset is None:
        dataset, val_dataset = _split(dataset, config.val_ratio, config.seed)

    loaders = {
        "train": DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.loader_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
    }
    if val_dataset is not None:
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.loader_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_unet(config.model).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    flow = FlowMatching(time_scale=config.time_scale, sigma_min=config.sigma_min)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = max(1, config.epochs * len(loaders["train"]) // config.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, _warmup_cosine(total_steps, config.warmup_steps)
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def batch_loss(batch: dict) -> torch.Tensor:
        obstacle = batch["obstacle"].to(device, non_blocking=True)
        x0 = batch["target"].to(device, non_blocking=True)
        start = endpoint_heatmaps(batch["start_pixel"].to(device), grid.size, config.endpoint_sigma)
        goal = endpoint_heatmaps(batch["goal_pixel"].to(device), grid.size, config.endpoint_sigma)
        noise = torch.randn_like(x0)
        t = flow.sample_t(x0.shape[0], device, config.t_mode)
        x_t, velocity = flow.interpolate(x0, noise, t)
        with torch.amp.autocast("cuda", enabled=use_amp):
            return F.mse_loss(model(x_t, obstacle, start, goal, t * flow.time_scale), velocity)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_loss = float("inf")
    stale_epochs = 0
    began = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        running = 0.0
        for index, batch in enumerate(loaders["train"]):
            loss = batch_loss(batch) / config.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item() * config.grad_accum_steps
            if (index + 1) % config.grad_accum_steps:
                continue
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters(), strict=False):
                    target.mul_(config.ema_decay).add_(source, alpha=1.0 - config.ema_decay)
                for target, source in zip(ema.buffers(), model.buffers(), strict=False):
                    target.copy_(source)

        train_loss = running / max(len(loaders["train"]), 1)
        message = f"epoch {epoch:3d}  train {train_loss:.6f}"

        if "val" in loaders and epoch % config.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = sum(batch_loss(b).item() for b in loaders["val"]) / len(loaders["val"])
            message += f"  val {val_loss:.6f}"
            if val_loss < best_loss:
                best_loss = val_loss
                stale_epochs = 0
                torch.save(
                    {
                        "model": ema.state_dict(),
                        "model_config": config.model,
                        "flow_config": {
                            "time_scale": config.time_scale,
                            "sigma_min": config.sigma_min,
                        },
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "train_config": asdict(config) | {"output_dir": str(config.output_dir)},
                    },
                    best_path,
                )
                message += "  *"
            else:
                stale_epochs += 1

        print(f"{message}  [{time.perf_counter() - began:.0f}s]", flush=True)
        if stale_epochs >= config.early_stop_patience:
            print(f"early stop after {epoch} epochs", flush=True)
            break

    return best_path
