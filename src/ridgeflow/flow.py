from __future__ import annotations

import torch


class FlowMatching:
    """Rectified flow on the straight path ``x_t = (1 - t) x0 + t eps``.

    ``t = 0`` is data and ``t = 1`` is noise, so the velocity the model regresses is the
    constant ``eps - x0`` and sampling counts ``t`` down. ``time_scale`` maps ``t`` onto
    the range the sinusoidal embedding expects and only has to agree between training
    and sampling.
    """

    def __init__(self, time_scale: float = 1000.0, sigma_min: float = 0.0) -> None:
        self.time_scale = float(time_scale)
        self.sigma_min = float(sigma_min)

    def sample_t(self, n: int, device, mode: str = "uniform") -> torch.Tensor:
        if mode == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device))
        return torch.rand(n, device=device)

    def interpolate(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shaped = t.view(-1, *([1] * (x0.dim() - 1)))
        if self.sigma_min > 0.0:
            alpha = 1.0 - (1.0 - self.sigma_min) * shaped
            return alpha * x0 + shaped * noise, noise - (1.0 - self.sigma_min) * x0
        return (1.0 - shaped) * x0 + shaped * noise, noise - x0

    @torch.no_grad()
    def sample(
        self,
        model,
        obstacle: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        steps: int = 4,
        eta: float = 0.5,
        rho: float = 1.0,
        seed: int | None = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Euler integration from noise to data, optionally re-noising after each step.

        ``eta`` mixes the deterministic Euler step with an ancestral one that predicts
        ``x0 = x - t v`` and re-noises it to the next level. It is the escape route from
        the between-modes region the averaged velocity points into at high ``t``.
        """
        shape = (obstacle.shape[0], 1, obstacle.shape[-2], obstacle.shape[-1])
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(shape, device=obstacle.device)
        schedule = [(1.0 - i / steps) ** rho for i in range(steps + 1)]
        for i in range(steps):
            t, t_next = schedule[i], schedule[i + 1]
            batched = torch.full((shape[0],), t * self.time_scale, device=obstacle.device)
            velocity = model(x, obstacle, start, goal, batched)
            deterministic = x - velocity * (t - t_next)
            if eta > 0.0 and t_next > 0.0:
                x0 = x - t * velocity
                stochastic = (1.0 - t_next) * x0 + t_next * torch.randn_like(x)
                x = (1.0 - eta) * deterministic + eta * stochastic
            else:
                x = deterministic
        return x.clamp(0.0, 1.0) if clamp else x
