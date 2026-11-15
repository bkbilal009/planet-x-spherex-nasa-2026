"""Read SPHEREx Level-2 spectral image FITS files (full frames or IRSA cutouts).

SPHEREx spectral-image MEF layout (SPHEREx Explanatory Supplement):
    IMAGE     calibrated surface brightness [MJy/sr], celestial WCS
    FLAGS     per-pixel bitmask (bit names in MP_* header keywords)
    VARIANCE  per-pixel variance [(MJy/sr)^2]
    ZODI      modelled zodiacal-light background [MJy/sr]
    PSF       PSF cube
    WCS-WAVE  wavelength lookup table (alternate WCS 'W' on IMAGE)
Every extension except IMAGE is treated as optional.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from .config import FALLBACK_BAD_FLAG_NAMES


@contextlib.contextmanager
def quiet_astropy():
    """Silence astropy INFO logging (e.g. the SIP-without-'-SIP' CTYPE notice).

    SPHEREx headers carry SIP distortion with plain TAN CTYPEs; applying the
    SIP terms is correct for SPHEREx, so the notice is noise for users.
    """
    from astropy import log
    old = log.level
    log.setLevel("WARNING")
    try:
        yield
    finally:
        log.setLevel(old)


class FitsReadError(RuntimeError):
    """Raised when a file is not a usable SPHEREx image."""


@dataclass
class Observation:
    image: np.ndarray                 # float64, NaN = invalid
    wcs: WCS                          # 2-D celestial WCS
    variance: np.ndarray | None = None
    bad_mask: np.ndarray | None = None   # True = do not trust pixel
    zodi: np.ndarray | None = None
    wavelength_um: float | None = None   # wavelength at image centre
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape  # type: ignore[return-value]

    @property
    def has_variance(self) -> bool:
        return self.variance is not None


def _bad_bits_from_header(header: fits.Header) -> int:
    """Build a bitmask of 'bad' FLAGS bits from MP_<NAME> = bit keywords."""
    mask = 0
    for key, value in header.items():
        if not key.startswith("MP_"):
            continue
        name = key[3:]
        if name in FALLBACK_BAD_FLAG_NAMES and isinstance(value, int):
            mask |= 1 << value
    return mask


def _central_wavelength(hdul: fits.HDUList, header: fits.Header) -> float | None:
    """Evaluate the spectral WCS ('W') at the image centre, if available."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wave_wcs = WCS(header=header, fobj=hdul, key="W")
        ny, nx = header["NAXIS2"], header["NAXIS1"]
        # Two-axis lookup table: (x, y) -> (wavelength, bandwidth)
        out = wave_wcs.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2)
        wl = float(np.atleast_1d(out[0] if isinstance(out, (tuple, list)) else out)[0])
        return wl if np.isfinite(wl) else None
    except Exception:  # noqa: BLE001 - optional metadata only
        return None


def read_spherex_fits(path: str | Path, meta: dict[str, Any] | None = None,
                      subtract_zodi: bool = True) -> Observation:
    """Load a SPHEREx image (+ optional VARIANCE/FLAGS/ZODI) into an Observation.

    Invalid pixels are set to NaN (never median-imputed) so they can be
    excluded from the difference rather than creating fake residuals.
    """
    path = Path(path)
    if not path.is_file():
        raise FitsReadError(f"File not found: {path}")
    try:
        hdul = fits.open(path, memmap=False)
    except Exception as exc:  # noqa: BLE001
        raise FitsReadError(f"Not a valid FITS file: {path.name} ({exc})") from exc

    with hdul:
        names = [h.name.upper() for h in hdul]
        if "IMAGE" in names:
            img_hdu = hdul["IMAGE"]
        else:  # generic fallback: first HDU with 2-D data
            img_hdu = next((h for h in hdul if h.data is not None
                            and getattr(h.data, "ndim", 0) == 2), None)
        if img_hdu is None or img_hdu.data is None:
            raise FitsReadError(f"No 2-D IMAGE extension in {path.name}")

        header = img_hdu.header
        image = np.array(img_hdu.data, dtype=np.float64)
        with quiet_astropy(), warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header).celestial
        if not wcs.has_celestial:
            raise FitsReadError(f"{path.name} has no celestial WCS")

        def _opt(name: str) -> np.ndarray | None:
            if name in names and hdul[name].data is not None \
                    and hdul[name].data.shape == image.shape:
                return np.array(hdul[name].data)
            return None

        variance = _opt("VARIANCE")
        zodi = _opt("ZODI")
        flags = _opt("FLAGS")

        bad = ~np.isfinite(image)
        if flags is not None:
            bad_bits = _bad_bits_from_header(hdul["FLAGS"].header)
            if bad_bits:
                bad |= (flags.astype(np.int64) & bad_bits) != 0
        if variance is not None:
            variance = variance.astype(np.float64)
            bad |= ~np.isfinite(variance) | (variance <= 0)

        if subtract_zodi and zodi is not None:
            # Zodiacal light changes with solar elongation between epochs;
            # removing the model keeps it from dominating the difference.
            image = image - zodi.astype(np.float64)

        image[bad] = np.nan
        with quiet_astropy():
            wavelength = _central_wavelength(hdul, header)

    info = dict(meta or {})
    info.setdefault("file", path.name)
    info["zodi_subtracted"] = bool(subtract_zodi and zodi is not None)
    info["extensions"] = names
    for key in ("OBSID", "DATE-OBS", "MJD-OBS", "DETECTOR", "BUNIT"):
        if key in header:
            info.setdefault(key.lower().replace("-", "_"), header[key])

    return Observation(image=image, wcs=wcs, variance=variance, bad_mask=bad,
                       zodi=zodi, wavelength_um=wavelength, meta=info)
