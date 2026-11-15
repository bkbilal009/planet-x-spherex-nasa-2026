"""WCS-based alignment of two observations onto a common pixel grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS

from .fits_processor import Observation


@dataclass
class AlignedPair:
    earlier: Observation          # on the reference grid (unchanged)
    later: Observation            # reprojected onto the earlier grid
    footprint: np.ndarray         # True where both images have valid data
    residual_shift_pix: tuple[float, float] | None = None  # (dy, dx)


def _fill_nearest(arr: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace ``bad`` pixels with the value of the nearest good pixel."""
    from scipy import ndimage
    if not bad.any():
        return arr
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def reproject_observation(obs: Observation, target_wcs: WCS,
                          shape: tuple[int, int]) -> Observation:
    """Reproject image, variance and bad-pixel mask onto ``target_wcs``.

    The image uses bicubic interpolation: SPHEREx PSFs are undersampled (~6"
    pixels) and bilinear interpolation blurs only the resampled epoch, leaving
    strong residuals on bright static stars (measured ~30 sigma vs ~4 sigma for
    bicubic in the unit tests). The variance is interpolated bilinearly
    (bicubic can ring negative) - an approximation, since resampling
    correlates noise; this is reported alongside the significance map.
    """
    from reproject import reproject_interp
    from scipy import ndimage

    bad_in = ~np.isfinite(obs.image)
    if obs.bad_mask is not None:
        bad_in |= obs.bad_mask
    if bad_in.all():
        raise ValueError("Observation has no valid pixels.")
    # Spline interpolation prefilters the whole array, so a single NaN would
    # poison every output pixel. Fill bad pixels from their nearest valid
    # neighbour (a constant fill leaves a sharp step inside saturated stars that
    # makes the bicubic spline ring), interpolate, then re-mask below.
    filled = _fill_nearest(obs.image, bad_in)
    image, foot = reproject_interp((filled, obs.wcs), target_wcs,
                                   shape_out=shape, order="bicubic")

    # Output pixels whose interpolation stencil touched a bad input pixel.
    bad_frac, _ = reproject_interp((bad_in.astype(float), obs.wcs), target_wcs,
                                   shape_out=shape, order="bilinear")
    bad = ~np.isfinite(bad_frac) | (bad_frac > 0) | (foot <= 0) | ~np.isfinite(image)
    bad = ndimage.binary_dilation(bad, iterations=1)   # bicubic stencil is 4x4

    variance = None
    if obs.variance is not None:
        var_in = _fill_nearest(obs.variance, bad_in | ~np.isfinite(obs.variance))
        variance, _ = reproject_interp((var_in, obs.wcs), target_wcs,
                                       shape_out=shape, order="bilinear")
        bad |= ~np.isfinite(variance)
        variance = np.where(bad, np.nan, variance)
    image = np.where(bad, np.nan, image)

    return Observation(image=image, wcs=target_wcs, variance=variance,
                       bad_mask=bad, zodi=None,
                       wavelength_um=obs.wavelength_um,
                       meta={**obs.meta, "reprojected": True})


def estimate_residual_shift(a: np.ndarray, b: np.ndarray,
                            upsample: int = 10) -> tuple[float, float] | None:
    """Estimate the (dy, dx) pixel offset of ``b`` relative to ``a``.

    Positive dx means features in ``b`` sit at larger x than in ``a``.

    FFT cross-correlation with parabolic sub-pixel refinement. Used only as a
    diagnostic: a shift of >~0.3 px means stars will leave dipole residuals
    that must not be mistaken for moving objects.
    """
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100:
        return None
    a0 = np.where(valid, a - np.nanmedian(a[valid]), 0.0)
    b0 = np.where(valid, b - np.nanmedian(b[valid]), 0.0)
    # Clip extreme values so a single bright star does not dominate.
    for arr in (a0, b0):
        lim = np.percentile(np.abs(arr[valid]), 99.5)
        np.clip(arr, -lim, lim, out=arr)
    cc = np.fft.ifft2(np.fft.fft2(a0) * np.conj(np.fft.fft2(b0))).real
    cc = np.fft.fftshift(cc)
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    cy, cx = cc.shape[0] // 2, cc.shape[1] // 2

    def _refine(m1: float, c0: float, p1: float) -> float:
        denom = m1 - 2 * c0 + p1
        return 0.0 if denom == 0 else 0.5 * (m1 - p1) / denom

    dy = dx = 0.0
    if 0 < py < cc.shape[0] - 1:
        dy = _refine(cc[py - 1, px], cc[py, px], cc[py + 1, px])
    if 0 < px < cc.shape[1] - 1:
        dx = _refine(cc[py, px - 1], cc[py, px], cc[py, px + 1])
    return float(-(py - cy + dy)), float(-(px - cx + dx))


def align_pair(earlier: Observation, later: Observation) -> AlignedPair:
    """Reproject ``later`` onto ``earlier``'s pixel grid."""
    later_r = reproject_observation(later, earlier.wcs, earlier.shape)
    bad_e = earlier.bad_mask if earlier.bad_mask is not None \
        else ~np.isfinite(earlier.image)
    footprint = ~bad_e & ~later_r.bad_mask
    shift = estimate_residual_shift(np.where(footprint, earlier.image, np.nan),
                                    np.where(footprint, later_r.image, np.nan))
    return AlignedPair(earlier=earlier, later=later_r, footprint=footprint,
                       residual_shift_pix=shift)
