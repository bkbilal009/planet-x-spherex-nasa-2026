"""Target coordinates and proper-motion propagation."""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from .config import Target


def position_at(target: Target, when: Time | float | None) -> SkyCoord:
    """Linear proper-motion propagation of ``target`` to time ``when``.

    ``when`` may be an astropy Time or an MJD. Linear propagation on the
    tangent plane is accurate to well under a SPHEREx pixel for decades even
    for Barnard's Star (radial-velocity/perspective terms are ~mas/yr^2).
    """
    if when is None:
        return SkyCoord(target.ra_deg * u.deg, target.dec_deg * u.deg)
    t = when if isinstance(when, Time) else Time(float(when), format="mjd")
    dt_yr = t.jyear - target.ref_epoch
    dec = target.dec_deg + target.pm_dec_mas_yr * dt_yr / 3.6e6
    ra = target.ra_deg + (target.pm_ra_cosdec_mas_yr * dt_yr / 3.6e6) \
        / np.cos(np.radians(target.dec_deg))
    return SkyCoord((ra % 360) * u.deg, dec * u.deg)


def custom_target(ra_deg: float, dec_deg: float, name: str = "Custom position") -> Target:
    return Target(name, name, float(ra_deg), float(dec_deg), note="User-entered coordinates.")


def parse_coordinates(ra_text: str, dec_text: str) -> SkyCoord:
    """Accept decimal degrees or sexagesimal (``08h55m10.8s`` / ``08:55:10.8``)."""
    ra_text, dec_text = ra_text.strip(), dec_text.strip()
    try:
        return SkyCoord(float(ra_text) * u.deg, float(dec_text) * u.deg)
    except ValueError:
        return SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg))
