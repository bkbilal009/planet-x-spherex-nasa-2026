"""Pure-NumPy robust statistics.

These replace astropy.stats / astropy.visualization helpers, whose compiled
extension (_fast_sigma_clip) can be blocked by Windows Application Control on
managed machines. Behaviour matches the astropy defaults closely.
"""
from __future__ import annotations

import numpy as np


def sigma_clipped_stats(values: np.ndarray, sigma: float = 3.0,
                        maxiters: int = 5) -> tuple[float, float, float]:
    """(mean, median, std) after iterative median-centred sigma clipping."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    for _ in range(maxiters):
        med, std = np.median(v), np.std(v)
        keep = np.abs(v - med) <= sigma * std
        if keep.all() or keep.sum() < 3:
            break
        v = v[keep]
    return float(np.mean(v)), float(np.median(v)), float(np.std(v))


def mad_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def zscale_limits(values: np.ndarray, n_samples: int = 1000, contrast: float = 0.25,
                  max_reject: float = 0.5, min_npixels: int = 5, krej: float = 2.5,
                  max_iterations: int = 5) -> tuple[float, float]:
    """IRAF ZScale display limits (same algorithm as astropy ZScaleInterval)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    stride = int(max(1.0, v.size / n_samples))
    samples = np.sort(v[::stride][:n_samples])
    npix = samples.size
    vmin, vmax = samples[0], samples[-1]
    x = np.arange(npix)
    ngood = npix
    last_ngood = npix + 1
    badpix = np.zeros(npix, dtype=bool)
    minpix = max(min_npixels, int(npix * max_reject))
    slope = intercept = 0.0
    for _ in range(max_iterations):
        if ngood >= last_ngood or ngood < minpix:
            break
        fit = np.polyfit(x[~badpix], samples[~badpix], deg=1)
        slope, intercept = fit
        flat = samples - (slope * x + intercept)
        threshold = krej * flat[~badpix].std()
        badpix[(flat < -threshold) | (flat > threshold)] = True
        # Grow rejected pixels by one on each side.
        badpix = np.convolve(badpix, np.ones(3, bool), mode="same").astype(bool)
        last_ngood, ngood = ngood, int((~badpix).sum())
    if ngood >= minpix:
        if contrast > 0:
            slope /= contrast
        center = (npix - 1) // 2
        median = np.median(samples)
        vmin = max(vmin, median - (center - 1) * slope)
        vmax = min(vmax, median + (npix - center) * slope)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def symmetric_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    """Symmetric colour limit for diverging maps (so zero sits at the centre)."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, percentile))
    return lim if lim > 0 else 1.0
