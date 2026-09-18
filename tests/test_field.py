from __future__ import annotations

import numpy as np
import pytest

from ridgeflow.field import EPS, bilinear_sample, gaussian_blur, squared_distance_field


def _ridge(size: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(float(size))
    distance = np.abs(axis - size / 2.0)[:, None] * np.ones((1, size))
    return np.exp(-(distance**2) / (2.0 * sigma**2)), distance


def test_inversion_recovers_distance() -> None:
    sigma = 1.5
    heatmap, distance = _ridge(64, sigma)
    recovered = np.sqrt(squared_distance_field(heatmap, sigma, blur_sigma=0.0))
    near = distance <= 5.0
    assert np.allclose(recovered[near], distance[near], atol=1e-3)


def test_blur_widens_the_effective_sigma() -> None:
    sigma, blur = 1.5, 2.0
    heatmap, distance = _ridge(96, sigma)
    recovered = squared_distance_field(heatmap, sigma, blur_sigma=blur)
    near = distance <= 5.0
    assert np.allclose(recovered[near], distance[near] ** 2, atol=0.2)


def test_inversion_saturates_beyond_the_floor() -> None:
    """The log of a clipped heatmap cannot report a distance past its own floor."""
    sigma = 1.5
    heatmap, distance = _ridge(64, sigma)
    ceiling = -2.0 * sigma**2 * np.log(EPS)
    recovered = squared_distance_field(heatmap, sigma, blur_sigma=0.0)
    assert recovered.max() == pytest.approx(ceiling, rel=1e-9)
    assert distance.max() ** 2 > ceiling


def test_field_is_monotone_away_from_the_ridge() -> None:
    heatmap, _ = _ridge(96, 1.5)
    column = squared_distance_field(heatmap, 1.5, blur_sigma=1.5)[48:60, 48]
    assert np.all(np.diff(column) > 0.0)


def test_blur_preserves_mass() -> None:
    field = np.zeros((32, 32))
    field[16, 16] = 1.0
    assert gaussian_blur(field, 2.0).sum() == pytest.approx(1.0, rel=1e-6)


def test_blur_is_a_noop_at_zero_sigma() -> None:
    field = np.random.default_rng(0).random((16, 16))
    assert gaussian_blur(field, 0.0) is field


def test_bilinear_sample_interpolates() -> None:
    field = np.arange(16.0).reshape(4, 4)
    assert bilinear_sample(field, np.array([1.5]), np.array([0.0]))[0] == pytest.approx(6.0)
    assert bilinear_sample(field, np.array([0.0]), np.array([0.5]))[0] == pytest.approx(0.5)


def test_sampling_clamps_outside_the_grid() -> None:
    field = np.arange(16.0).reshape(4, 4)
    assert bilinear_sample(field, np.array([99.0]), np.array([-5.0]))[0] == pytest.approx(12.0)
