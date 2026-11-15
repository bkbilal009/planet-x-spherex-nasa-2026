"""Background/photometric normalisation, image differencing and significance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from .utils import mad_std, sigma_clipped_stats

from .image_alignment import AlignedPair


@dataclass
class DifferenceResult:
    difference: np.ndarray            # later - earlier [image units]
    significance: np.ndarray          # difference / sigma
    significance_kind: str            # "measured" (VARIANCE) or "estimated" (MAD)
    valid: np.ndarray                 # pixels used
    earlier: np.ndarray               # normalised earlier image
    later: np.ndarray                 # normalised later image
    notes: list[str] = field(default_factory=list)
    flux_scale: float = 1.0
    noise_sigma: float = float("nan")  # robust std of the difference


def subtract_background(image: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove a constant sky level estimated with 3-sigma clipping."""
    _, median, _ = sigma_clipped_stats(image[valid], sigma=3.0, maxiters=5)
    return image - median, float(median)


def fit_flux_scale(earlier: np.ndarray, later: np.ndarray, valid: np.ndarray,
                   var_e: np.ndarray | None = None, var_l: np.ndarray | None = None,
                   snr_min: float = 10.0) -> float | None:
    """Robust ratio later/earlier over bright pixels (median of ratios).

    SPHEREx images are already flux-calibrated, so this is optional and only
    corrects residual calibration offsets. Returns None if too few bright
    pixels exist to constrain it.
    """
    if var_e is not None:
        snr = earlier / np.sqrt(np.where(var_e > 0, var_e, np.nan))
    else:
        _, _, std = sigma_clipped_stats(earlier[valid])
        snr = earlier / std if std > 0 else np.zeros_like(earlier)
    use = valid & np.isfinite(snr) & (snr > snr_min) & (later > 0)
    if use.sum() < 20:
        return None
    ratio = later[use] / earlier[use]
    _, med, _ = sigma_clipped_stats(ratio, sigma=3.0)
    return float(med) if np.isfinite(med) and med > 0 else None


def gaussian_kernel(sigma_pix: float) -> np.ndarray:
    half = max(1, int(np.ceil(4 * sigma_pix)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_pix ** 2))
    return k / k.sum()


def smooth(image: np.ndarray, variance: np.ndarray | None,
           sigma_pix: float) -> tuple[np.ndarray, np.ndarray | None]:
    """NaN-aware Gaussian smoothing with correct variance propagation.

    For a normalised kernel k, Var(k * x) = (k^2) * Var(x) for independent
    pixels. Used to degrade both epochs to a common (broader) PSF, which
    suppresses residuals caused by small PSF differences between epochs.
    """
    from astropy.convolution import convolve
    k = gaussian_kernel(sigma_pix)
    out = convolve(image, k, boundary="extend", nan_treatment="interpolate",
                   preserve_nan=True)
    var_out = None
    if variance is not None:
        var_out = convolve(variance, k ** 2, boundary="extend",
                           nan_treatment="interpolate", preserve_nan=True,
                           normalize_kernel=False)
    return out, var_out


def compute_difference(pair: AlignedPair, *, remove_background: bool = True,
                       match_flux: bool = False, psf_sigma_pix: float = 0.0
                       ) -> DifferenceResult:
    """Difference = Later - Earlier on the common grid, with significance.

    Significance = (L - E) / sqrt(Var_E + Var_L) where VARIANCE extensions
    exist ("measured"). Otherwise sigma is the robust (MAD) std of the
    difference image itself ("estimated" - spatially uniform approximation).
    """
    notes: list[str] = []
    e = pair.earlier.image.copy()
    l = pair.later.image.copy()
    var_e = None if pair.earlier.variance is None else pair.earlier.variance.copy()
    var_l = None if pair.later.variance is None else pair.later.variance.copy()
    valid = pair.footprint & np.isfinite(e) & np.isfinite(l)
    if valid.sum() < 10:
        raise ValueError("The two observations do not overlap on the sky.")

    if psf_sigma_pix > 0:
        e, var_e = smooth(np.where(valid, e, np.nan), var_e, psf_sigma_pix)
        l, var_l = smooth(np.where(valid, l, np.nan), var_l, psf_sigma_pix)
        valid &= np.isfinite(e) & np.isfinite(l)
        notes.append(f"Both epochs smoothed with a Gaussian (sigma={psf_sigma_pix:.2f} px) "
                     "to a common PSF.")

    if remove_background:
        e, be = subtract_background(e, valid)
        l, bl = subtract_background(l, valid)
        notes.append(f"Residual sky levels removed: earlier {be:.4g}, later {bl:.4g}.")

    scale = 1.0
    if match_flux:
        fitted = fit_flux_scale(e, l, valid, var_e, var_l)
        if fitted is None:
            notes.append("Flux scale not fitted (too few bright pixels); using 1.0.")
        else:
            scale = fitted
            l = l / scale
            if var_l is not None:
                var_l = var_l / scale ** 2
            notes.append(f"Later image divided by fitted flux scale {scale:.4f}.")

    diff = np.where(valid, l - e, np.nan)

    if var_e is not None and var_l is not None:
        total = var_e + var_l
        # A non-positive variance in either epoch is unphysical -> pixel invalid.
        ok = valid & np.isfinite(total) & (var_e > 0) & (var_l > 0)
        sig = np.full_like(diff, np.nan)
        sig[ok] = diff[ok] / np.sqrt(total[ok])
        kind = "measured"
        valid = ok
        if pair.later.meta.get("reprojected"):
            notes.append("Later-epoch variance was interpolated during reprojection; "
                         "significances are slightly optimistic for the resampled image.")
    else:
        noise = mad_std(diff[valid])
        sig = diff / noise if noise > 0 else np.full_like(diff, np.nan)
        kind = "estimated"
        notes.append("No VARIANCE extension: significance uses a uniform noise estimate "
                     "(MAD of the difference) - approximate only.")

    return DifferenceResult(difference=diff, significance=sig, significance_kind=kind,
                            valid=valid, earlier=np.where(valid, e, np.nan),
                            later=np.where(valid, l, np.nan), notes=notes,
                            flux_scale=scale, noise_sigma=mad_std(diff[valid]))

