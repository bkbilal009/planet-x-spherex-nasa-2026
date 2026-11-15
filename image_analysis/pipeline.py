"""End-to-end change detection for two observations (UI-independent)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .candidate_detection import detect_candidates
from .fits_processor import Observation
from .image_alignment import AlignedPair, align_pair
from .image_subtraction import DifferenceResult, compute_difference
from .utils import mad_std


@dataclass
class ChangeDetectionResult:
    pair: AlignedPair
    diff: DifferenceResult
    candidates: pd.DataFrame
    threshold: float
    noise_scale: float = 1.0     # empirical rescale applied to significance (>= 1)


def empirical_noise_scale(sig: np.ndarray, valid: np.ndarray, threshold: float,
                          iterations: int = 3) -> float:
    """Robust width of the significance map outside candidate regions.

    For a correct noise model this is ~1. Real SPHEREx differences often show
    a larger width (noise correlated by resampling, residual zodiacal/sky
    structure, PSF wings of bright variables), which would make a fixed
    sigma threshold too permissive. Iteratively masks |sig| >= threshold
    (dilated) so real changes do not inflate the estimate.
    """
    scale = 1.0
    for _ in range(iterations):
        hot = ndimage.binary_dilation(valid & (np.abs(np.nan_to_num(sig / scale)) >= threshold),
                                      iterations=3)
        use = valid & ~hot
        if use.sum() < 50:
            break
        scale = max(1.0, mad_std(sig[use]))
    return float(scale)


def order_by_time(a: Observation, b: Observation) -> tuple[Observation, Observation]:
    ta, tb = a.meta.get("t_min"), b.meta.get("t_min")
    if ta is not None and tb is not None and float(ta) > float(tb):
        return b, a
    return a, b


def run_change_detection(obs_a: Observation, obs_b: Observation, *,
                         threshold: float = 5.0, min_pixels: int = 2,
                         remove_background: bool = True, match_flux: bool = False,
                         psf_sigma_pix: float = 0.0,
                         rescale_noise: bool = True) -> ChangeDetectionResult:
    earlier, later = order_by_time(obs_a, obs_b)
    pair = align_pair(earlier, later)
    diff = compute_difference(pair, remove_background=remove_background,
                              match_flux=match_flux, psf_sigma_pix=psf_sigma_pix)
    scale = 1.0
    if rescale_noise:
        scale = empirical_noise_scale(diff.significance, diff.valid, threshold)
        if scale > 1.05:
            diff.significance = diff.significance / scale
            diff.notes.append(f"Significance divided by empirical noise factor {scale:.2f} "
                              "(robust width of the map outside candidate regions).")
    cands = detect_candidates(diff.significance, diff.difference, diff.valid,
                              wcs=pair.earlier.wcs, threshold=threshold,
                              min_pixels=min_pixels)
    shift = pair.residual_shift_pix
    if shift is not None and max(abs(shift[0]), abs(shift[1])) > 0.3:
        diff.notes.append(f"Residual offset after WCS alignment ~({shift[1]:+.2f}, {shift[0]:+.2f}) px "
                          "(x, y): stars may leave dipole residuals - treat dipoles with caution.")
    return ChangeDetectionResult(pair=pair, diff=diff, candidates=cands, threshold=threshold,
                                 noise_scale=scale)
