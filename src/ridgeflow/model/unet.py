from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    return min(8, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 else emb


class TimeMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.conv2(F.silu(h)) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, downsample: bool
    ) -> None:
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_dim)
        self.down = (
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res2(self.res1(x, t_emb), t_emb)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, upsample: bool
    ) -> None:
        super().__init__()
        self.res1 = ResBlock(in_channels + skip_channels, out_channels, time_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_dim)
        self.up = (
            nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
            if upsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.up(self.res2(self.res1(x, t_emb), t_emb))


class ConditionalUNet(nn.Module):
    """Predicts a velocity field over the ridge heatmap, conditioned on map and endpoints."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        for index, out in enumerate(channels):
            self.downs.append(DownBlock(current, out, time_emb_dim, index < len(channels) - 1))
            current = out

        self.mid1 = ResBlock(current, current, time_emb_dim)
        self.mid2 = ResBlock(current, current, time_emb_dim)

        self.ups = nn.ModuleList()
        for index, skip in enumerate(reversed(channels)):
            self.ups.append(
                UpBlock(current, skip, skip, time_emb_dim, index < len(channels) - 1)
            )
            current = skip

        self.final = nn.Sequential(
            nn.GroupNorm(_groups(current), current),
            nn.SiLU(),
            nn.Conv2d(current, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        obstacle: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.init_conv(torch.cat([x_t, obstacle, start, goal], dim=1))
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.mid2(self.mid1(x, t_emb), t_emb)
        for up, skip in zip(self.ups, reversed(skips), strict=False):
            x = up(x, skip, t_emb)
        return self.final(x)


def build_unet(config: dict) -> ConditionalUNet:
    return ConditionalUNet(
        in_channels=int(config.get("in_channels", 4)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 64)),
        time_emb_dim=int(config.get("time_emb_dim", 256)),
        channel_mults=tuple(config.get("channel_mults", (1, 2, 4))),
    )
