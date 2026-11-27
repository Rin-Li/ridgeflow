from __future__ import annotations

import torch


class FlowMatching:
    """Rectified flow on ``x_t = (1 - t) x0 + t eps``; the model regresses ``eps - x0``."""

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

    @staticmethod
    def noise(shape, steps: int, eta: float, device, seed: int | None = None):
        """The draws ``sample`` makes, in its order: the starting noise ``x`` and one draw per
        re-noised step, ``(steps - 1, *shape)`` (empty when ``eta`` is 0)."""
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(shape, device=device)
        kicks = torch.empty((steps - 1 if eta > 0.0 else 0, *shape), device=device)
        for kick in kicks:
            kick.normal_()
        return x, kicks

    def integrate(
        self, model, x, kicks, obstacle, start, goal, steps: int, eta: float, rho: float = 1.0
    ) -> torch.Tensor:
        """Euler steps from ``x`` at t = 1 to data, re-noised by ``eta`` with ``kicks[i]`` after
        every step but the last. Draws nothing, so it can be compiled as a whole."""
        schedule = [(1.0 - i / steps) ** rho for i in range(steps + 1)]
        for i in range(steps):
            t, t_next = schedule[i], schedule[i + 1]
            batched = torch.full((x.shape[0],), t * self.time_scale, device=x.device)
            velocity = model(x, obstacle, start, goal, batched).float()
            deterministic = x - velocity * (t - t_next)
            if eta > 0.0 and t_next > 0.0:
                stochastic = (1.0 - t_next) * (x - t * velocity) + t_next * kicks[i]
                x = (1.0 - eta) * deterministic + eta * stochastic
            else:
                x = deterministic
        return x

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
        """Euler steps from noise to data, re-noised by ``eta`` after each step."""
        shape = (obstacle.shape[0], 1, obstacle.shape[-2], obstacle.shape[-1])
        x, kicks = self.noise(shape, steps, eta, obstacle.device, seed)
        x = self.integrate(model, x, kicks, obstacle, start, goal, steps, eta, rho)
        return x.clamp(0.0, 1.0) if clamp else x
