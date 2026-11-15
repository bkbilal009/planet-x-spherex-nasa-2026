"""Group significant difference pixels into candidate change regions.

Candidates are *changes*, not classifications: a positive/negative pair
("dipole") is consistent with motion, but equally with residual misalignment
or PSF mismatch, and is labelled as such for the user to inspect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy import ndimage

from .config import (DEFAULT_DETECTION_THRESHOLD_SIGMA, DEFAULT_EDGE_MARGIN_PIX,
                     DEFAULT_MIN_CANDIDATE_PIXELS)

CANDIDATE_COLUMNS = ["id", "sign", "x", "y", "ra_deg", "dec_deg", "n_pix",
                     "peak_sigma", "sum_diff", "near_edge", "dipole_partner", "notes"]


def detect_candidates(significance: np.ndarray, difference: np.ndarray,
                      valid: np.ndarray, wcs: WCS | None = None,
                      threshold: float = DEFAULT_DETECTION_THRESHOLD_SIGMA,
                      min_pixels: int = DEFAULT_MIN_CANDIDATE_PIXELS,
                      edge_margin: int = DEFAULT_EDGE_MARGIN_PIX,
                      dipole_radius_pix: float = 4.0) -> pd.DataFrame:
    """Return one row per connected region with |significance| >= threshold.

    Brightening (sign=+1) and fading (sign=-1) are segmented separately using
    8-connectivity. Centroids are significance-weighted.
    """
    sig = np.where(valid & np.isfinite(significance), significance, 0.0)
    # Distance to the nearest invalid pixel - candidates close to masked
    # regions/edges are more likely artefacts.
    dist_invalid = ndimage.distance_transform_edt(np.pad(valid, 1, constant_values=False))[1:-1, 1:-1]
    structure = np.ones((3, 3), dtype=bool)
    rows: list[dict] = []
    for sign in (+1, -1):
        mask = (sign * sig) >= threshold
        labels, n = ndimage.label(mask, structure=structure)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(mask, labels, idx)
        weights = np.abs(sig)
        cy_cx = ndimage.center_of_mass(weights, labels, idx)
        peaks = ndimage.maximum(sign * sig, labels, idx)
        sums = ndimage.sum(np.nan_to_num(difference), labels, idx)
        min_dist = ndimage.minimum(dist_invalid, labels, idx)
        for size, (cy, cx), peak, s, md in zip(sizes, cy_cx, peaks, sums, min_dist):
            if size < min_pixels:
                continue
            rows.append(dict(sign=sign, x=float(cx), y=float(cy), n_pix=int(size),
                             peak_sigma=float(sign * peak), sum_diff=float(s),
                             near_edge=bool(md <= edge_margin)))

    df = pd.DataFrame(rows, columns=[c for c in CANDIDATE_COLUMNS if c not in
                                     ("id", "ra_deg", "dec_deg", "dipole_partner", "notes")])
    if df.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    df = df.reindex(df["peak_sigma"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))

    if wcs is not None and wcs.has_celestial:
        world = wcs.pixel_to_world(df["x"].to_numpy(), df["y"].to_numpy())
        df["ra_deg"], df["dec_deg"] = world.ra.deg, world.dec.deg
    else:
        df["ra_deg"], df["dec_deg"] = np.nan, np.nan

    # Pair opposite-sign regions closer than dipole_radius_pix.
    partner = [None] * len(df)
    pos = df[df.sign > 0]
    neg = df[df.sign < 0]
    for i, p in pos.iterrows():
        if neg.empty:
            break
        d = np.hypot(neg.x - p.x, neg.y - p.y)
        j = d.idxmin()
        if d[j] <= dipole_radius_pix:
            partner[i] = int(df.at[j, "id"])
            partner[j] = int(p.id)
    df["dipole_partner"] = pd.array(partner, dtype="Int64")

    notes = []
    for _, r in df.iterrows():
        n = ["brightening" if r.sign > 0 else "fading"]
        if not pd.isna(r.dipole_partner):
            n.append(f"dipole with #{r.dipole_partner} (motion or misalignment)")
        if r.near_edge:
            n.append("near masked pixels/edge")
        if r.n_pix == 1:
            n.append("single pixel")
        notes.append("; ".join(n))
    df["notes"] = notes
    return df[CANDIDATE_COLUMNS]


def candidate_cutout(arr: np.ndarray, x: float, y: float, half: int = 12) -> np.ndarray:
    """Small stamp around (x, y) padded with NaN at image edges."""
    xi, yi = int(round(x)), int(round(y))
    out = np.full((2 * half + 1, 2 * half + 1), np.nan)
    y0, y1 = max(0, yi - half), min(arr.shape[0], yi + half + 1)
    x0, x1 = max(0, xi - half), min(arr.shape[1], xi + half + 1)
    out[y0 - (yi - half):y1 - (yi - half), x0 - (xi - half):x1 - (xi - half)] = arr[y0:y1, x0:x1]
    return out
