from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ridgeflow.simulation import EpisodeResult, Outcome


@dataclass(frozen=True)
class Summary:
    """Aggregate of one condition; rates are fractions of all episodes."""

    episodes: int
    rates: dict[str, float]
    mean_path_length: float
    mean_inferences: float
    mean_waited: float
    staleness_px: float

    @property
    def arrived(self) -> float:
        return self.rates[Outcome.ARRIVED.value]

    def row(self, label: str) -> str:
        return (
            f"{label:<22} arrived={100 * self.arrived:6.2f}%  "
            f"hit={100 * self.rates[Outcome.HIT.value]:5.1f}%  "
            f"frozen={100 * self.rates[Outcome.FROZEN.value]:5.1f}%  "
            f"trapped={100 * self.rates[Outcome.TRAPPED.value]:4.1f}%  "
            f"len={self.mean_path_length:5.2f}  "
            f"stale={self.staleness_px:4.2f}px"
        )


def summarise(results: list[EpisodeResult]) -> Summary:
    counts = Counter(result.outcome.value for result in results)
    total = max(len(results), 1)
    return Summary(
        episodes=len(results),
        rates={outcome.value: counts.get(outcome.value, 0) / total for outcome in Outcome},
        mean_path_length=float(np.mean([r.path_length for r in results])),
        mean_inferences=float(np.mean([r.inferences for r in results])),
        mean_waited=float(np.mean([r.waited for r in results])),
        staleness_px=float(np.nanmean([r.staleness_px for r in results])),
    )
