"""Query and download real SPHEREx observations from NASA/IPAC IRSA.

Workflow (mirrors the official IRSA SPHEREx tutorials):
  1. SIA2 query of the SPHEREx spectral-image collection around a position
     (``Irsa.query_sia``). If SIA is unavailable, the same ObsCore metadata is
     queried through IRSA TAP (``ivoa.obscore``).
  2. Each row is one detector image (band SPHEREx-D1..D6) with an
     ``access_url`` to the Level-2 multi-extension FITS file.
  3. Download either the full 2040x2040 frame or an IRSA server-side cutout
     (``access_url?center=<ra>,<dec>deg&size=<n>arcmin``), which keeps all
     extensions (IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE).
No data are ever synthesised: every failure raises an ArchiveError with a
user-readable message.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from .config import (DEFAULT_COLLECTION, DOWNLOAD_TIMEOUT_S, SPHEREX_BANDS)
from .fits_processor import Observation, read_spherex_fits

# Columns shown to users (ObsCore names); missing ones are skipped.
METADATA_COLUMNS = [
    "obs_id", "obs_date_utc", "t_min", "t_max", "t_exptime",
    "energy_bandpassname", "detector", "em_min_um", "em_max_um",
    "lambda_target_um", "bandwidth_um",
    "s_ra", "s_dec", "calib_level", "dataproduct_subtype",
    "obs_collection", "access_estsize", "access_url",
]


class ArchiveError(RuntimeError):
    """Archive unavailable, empty result, or download failure."""


def _with_retries(func, attempts: int = 3, delay_s: float = 2.0):
    last: Exception | None = None
    for i in range(attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - network stack raises many types
            last = exc
            if i < attempts - 1:
                time.sleep(delay_s * (i + 1))
    raise last  # type: ignore[misc]


def _query_sia(coord: SkyCoord, radius: u.Quantity, collection: str) -> pd.DataFrame:
    from astroquery.ipac.irsa import Irsa
    table = Irsa.query_sia(pos=(coord, radius), collection=collection)
    return table.to_pandas()


def _query_tap(coord: SkyCoord, radius: u.Quantity, collection: str) -> pd.DataFrame:
    from astroquery.ipac.irsa import Irsa
    adql = (
        "SELECT * FROM ivoa.obscore "
        f"WHERE obs_collection = '{collection}' AND "
        f"INTERSECTS(CIRCLE('ICRS', {coord.ra.deg:.8f}, {coord.dec.deg:.8f}, "
        f"{radius.to_value(u.deg):.8f}), s_region) = 1"
    )
    return Irsa.query_tap(adql).to_table().to_pandas()


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:  # VOTable bytes -> str
        if df[col].dtype == object and len(df) and isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8")
    if "t_min" in df:
        df["obs_date_utc"] = Time(df["t_min"].to_numpy(dtype=float), format="mjd").iso
    if "energy_bandpassname" in df:
        df["detector"] = df["energy_bandpassname"].str.extract(r"D(\d)")[0]
    # ObsCore em_min/em_max are in metres.
    for src, dst in (("em_min", "em_min_um"), ("em_max", "em_max_um")):
        if src in df:
            df[dst] = df[src].astype(float) * 1e6
    if "dataproduct_subtype" in df:
        # Keep spectral images; drop any auxiliary products if present.
        sub = df["dataproduct_subtype"].astype(str).str.lower()
        keep = sub.str.contains("spectral_image") | (sub == "") | (sub == "nan")
        if keep.any():
            df = df[keep]
    df = df.sort_values("t_min") if "t_min" in df else df
    if "obs_id" in df and "energy_bandpassname" in df:
        df = df.drop_duplicates(subset=["obs_id", "energy_bandpassname"])
    return df.reset_index(drop=True)


def search_observations(coord: SkyCoord, radius_arcmin: float = 1.0,
                        collection: str = DEFAULT_COLLECTION,
                        bands: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Return (metadata, service_used) for SPHEREx images covering ``coord``.

    Raises ArchiveError if both SIA and TAP fail. An empty DataFrame (not an
    error) means the archive answered but has no coverage.
    """
    radius = max(radius_arcmin, 0.01) * u.arcmin
    errors: list[str] = []
    df = None
    service = ""
    for name, fn in (("SIA2", _query_sia), ("TAP (ivoa.obscore)", _query_tap)):
        try:
            df = _with_retries(lambda: fn(coord, radius, collection), attempts=2)
            service = name
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    if df is None:
        raise ArchiveError("IRSA archive query failed. The service may be temporarily "
                           "unavailable - please retry.\n" + "\n".join(errors))
    df = _tidy(df)
    if bands and "energy_bandpassname" in df:
        df = df[df["energy_bandpassname"].isin(bands)].reset_index(drop=True)
    return df, service


def display_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in METADATA_COLUMNS if c in df.columns]


def cutout_url(access_url: str, coord: SkyCoord, size_arcmin: float) -> str:
    """IRSA image cutout URL (IBE syntax, used by the SPHEREx cutout tutorial)."""
    sep = "&" if "?" in access_url else "?"
    return (f"{access_url}{sep}center={coord.ra.deg:.7f},{coord.dec.deg:.7f}deg"
            f"&size={size_arcmin:.4f}arcmin")


def download_fits(url: str, timeout: int = DOWNLOAD_TIMEOUT_S) -> Path:
    """Download via astropy's cache (no hard-coded local paths)."""
    from astropy.utils.data import download_file
    try:
        path = _with_retries(lambda: download_file(url, cache=True, timeout=timeout,
                                                   show_progress=False), attempts=4, delay_s=5.0)
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(f"Download failed: {type(exc).__name__}: {exc}") from exc
    return Path(path)


def fetch_observation(row: pd.Series, coord: SkyCoord | None = None,
                      size_arcmin: float | None = None,
                      subtract_zodi: bool = True) -> Observation:
    """Download one archive row (cutout if ``coord`` and ``size`` given) and read it."""
    url = str(row["access_url"])
    if coord is not None and size_arcmin:
        url = cutout_url(url, coord, size_arcmin)
    path = download_fits(url)
    meta = {k: row[k] for k in ("obs_id", "obs_date_utc", "t_min", "energy_bandpassname",
                                "detector", "em_min_um", "em_max_um", "obs_collection")
            if k in row.index and not (isinstance(row[k], float) and np.isnan(row[k]))}
    meta["source_url"] = url
    meta["cutout"] = bool(coord is not None and size_arcmin)
    return read_spherex_fits(path, meta=meta, subtract_zodi=subtract_zodi)


def valid_bands() -> tuple[str, ...]:
    return SPHEREX_BANDS


# ---------------------------------------------------------------------------
# Spectral matching
# ---------------------------------------------------------------------------
# SPHEREx uses linear variable filters: the wavelength seen by a pixel depends
# on its position on the detector. Two images in the same band can therefore
# sample a source at very different wavelengths (e.g. 1.64 vs 2.27 um for
# R Dra in D3). Temporal change detection should compare images whose
# wavelength *at the target* matches.

def wavelength_at(row: pd.Series, coord: SkyCoord,
                  size_arcmin: float = 1.0) -> tuple[float, float] | None:
    """(wavelength, bandwidth) in um at ``coord``, from a tiny IRSA cutout."""
    import warnings
    from astropy.io import fits
    from astropy.wcs import WCS
    from .fits_processor import quiet_astropy
    path = download_fits(cutout_url(str(row["access_url"]), coord, size_arcmin))
    with quiet_astropy(), warnings.catch_warnings(), fits.open(path) as hdul:
        warnings.simplefilter("ignore")
        hdr = hdul["IMAGE"].header
        x, y = WCS(hdr).celestial.world_to_pixel(coord)
        wl, bw = WCS(hdr, fobj=hdul, key="W").pixel_to_world_values(x, y)
    wl, bw = float(np.asarray(wl)), float(np.asarray(bw))
    return (wl, bw) if np.isfinite(wl) else None


def measure_wavelengths(df: pd.DataFrame, coord: SkyCoord, max_workers: int = 4,
                        progress=None) -> pd.DataFrame:
    """Add ``lambda_target_um`` / ``bandwidth_um`` columns (parallel downloads)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = df.copy()
    out["lambda_target_um"] = np.nan
    out["bandwidth_um"] = np.nan
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(wavelength_at, row, coord): idx for idx, row in df.iterrows()}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001 - unavailable cutout stays NaN
                res = None
            if res:
                out.at[idx, "lambda_target_um"], out.at[idx, "bandwidth_um"] = res
            done += 1
            if progress:
                progress(done / len(futs))
    return out


def suggest_pairs(df: pd.DataFrame, min_baseline_days: float = 1.0,
                  max_rel_dlambda: float = 0.01, top: int = 20) -> pd.DataFrame:
    """Spectrally matched pairs (|dlambda|/lambda <= tolerance), longest baseline first."""
    d = df.dropna(subset=["lambda_target_um", "t_min"]).reset_index()
    rows = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            a, b = d.iloc[i], d.iloc[j]
            if a["energy_bandpassname"] != b["energy_bandpassname"]:
                continue
            base = abs(float(b["t_min"]) - float(a["t_min"]))
            rel = abs(a["lambda_target_um"] - b["lambda_target_um"]) / a["lambda_target_um"]
            if base >= min_baseline_days and rel <= max_rel_dlambda:
                e, l = (a, b) if a["t_min"] <= b["t_min"] else (b, a)
                rows.append({"earlier_index": int(e["index"]), "later_index": int(l["index"]),
                             "earlier": f"{e['obs_id']} ({str(e['obs_date_utc'])[:10]})",
                             "later": f"{l['obs_id']} ({str(l['obs_date_utc'])[:10]})",
                             "band": a["energy_bandpassname"], "baseline_days": round(base, 2),
                             "lambda_um": round(float(e["lambda_target_um"]), 4),
                             "dlambda_rel_pct": round(100 * rel, 3)})
    out = pd.DataFrame(rows)
    return out.sort_values("baseline_days", ascending=False).head(top).reset_index(drop=True) \
        if not out.empty else out
