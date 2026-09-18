from __future__ import annotations

import numpy as np

EPS = 1e-6


def gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable reflect-padded Gaussian blur."""
    if sigma <= 0.0:
        return field
    radius = max(1, int(round(3.0 * sigma)))
    kernel = np.exp(-np.arange(-radius, radius + 1) ** 2 / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    def convolve(row: np.ndarray) -> np.ndarray:
        return np.convolve(np.pad(row, radius, mode="reflect"), kernel, "valid")

    blurred = np.apply_along_axis(convolve, 0, field)
    return np.apply_along_axis(convolve, 1, blurred)


def squared_distance_field(
    heatmap: np.ndarray, ridge_sigma: float = 1.5, blur_sigma: float = 1.5
) -> np.ndarray:
    """Invert a Gaussian ridge ``H = exp(-d^2 / 2 sigma^2)`` back into ``d^2``.

    Blurring widens the ridge, and variances add, so the effective sigma is
    ``hypot(ridge_sigma, blur_sigma)``. Peak normalisation makes the result
    independent of the heatmap's amplitude calibration.
    """
    blurred = gaussian_blur(np.asarray(heatmap, np.float64), blur_sigma)
    peak = max(float(blurred.max()), EPS)
    normalised = np.clip(blurred / peak, EPS, 1.0)
    sigma_effective = float(np.hypot(ridge_sigma, blur_sigma))
    return -2.0 * sigma_effective * sigma_effective * np.log(normalised)


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an ``[x, y]`` field at fractional pixel coordinates."""
    size_x, size_y = field.shape
    x = np.clip(x, 0.0, size_x - 1)
    y = np.clip(y, 0.0, size_y - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, size_x - 1)
    y1 = np.minimum(y0 + 1, size_y - 1)
    fx = x - x0
    fy = y - y0
    low = field[x0, y0] * (1.0 - fx) + field[x1, y0] * fx
    high = field[x0, y1] * (1.0 - fx) + field[x1, y1] * fx
    return low * (1.0 - fy) + high * fy
