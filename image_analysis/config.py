"""Configuration for the SPHEREx Image Analysis module.

Kept separate from processing logic so thresholds, archive identifiers and
challenge targets can be changed without touching algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
# IRSA SIA collection identifiers for SPHEREx spectral images (Level 2).
# Verified against Irsa.list_collections(servicetype="SIA", filter="spherex")
# - see image_analysis/README.md. Order = preference (first = default).
SPHEREX_COLLECTIONS: tuple[str, ...] = ("spherex_dr1", "spherex_qr3")
DEFAULT_COLLECTION = SPHEREX_COLLECTIONS[0]

# SPHEREx bands map 1:1 onto the six detector arrays.
SPHEREX_BANDS: tuple[str, ...] = tuple(f"SPHEREx-D{i}" for i in range(1, 7))

# Native pixel scale of SPHEREx (arcsec / pixel).
SPHEREX_PIXEL_SCALE_ARCSEC = 6.15

DOWNLOAD_TIMEOUT_S = 120
QUERY_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------
DEFAULT_CUTOUT_SIZE_ARCMIN = 10.0
DEFAULT_DETECTION_THRESHOLD_SIGMA = 5.0
DEFAULT_MIN_CANDIDATE_PIXELS = 2
DEFAULT_EDGE_MARGIN_PIX = 3

# SPHEREx FLAGS bits treated as "bad" for differencing. Per the SPHEREx
# Explanatory Supplement the FLAGS extension is a bitmask; the bits below are
# the hard-failure conditions (transient/overflow/saturation, dead/hot/
# non-functional pixels, missing data, nonlinear). Bits are read from the FITS
# header (MP_* keywords) when present, so this is only the fallback.
FALLBACK_BAD_FLAG_NAMES: tuple[str, ...] = (
    "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "DICHROIC",
    "MISSING_DATA", "HOT", "COLD", "FULLSAMPLE", "NONLINEAR", "BAD",
)


# ---------------------------------------------------------------------------
# Challenge targets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    name: str
    simbad_id: str
    ra_deg: float          # ICRS position at epoch ``ref_epoch``
    dec_deg: float
    pm_ra_cosdec_mas_yr: float = 0.0
    pm_dec_mas_yr: float = 0.0
    ref_epoch: float = 2000.0   # Julian year of the position
    note: str = ""
    source: str = ""


# Positions: ICRS, epoch J2000.0, as returned by SIMBAD (resolved through the
# CDS Sesame service on 2026-11-15). Proper motions from the same records.
# High proper-motion objects move arcsec/yr, so the app propagates positions to
# each observation date (image_analysis.targets.position_at).
CHALLENGE_TARGETS: tuple[Target, ...] = (
    Target("Barnard's Star", "Barnard's Star",
           269.4520769586, 4.6933649666, -801.551, 10362.394, 2000.0,
           "Highest proper-motion star (~10.4\"/yr): ~4.4' from its J2000 position by 2025.",
           "SIMBAD / Gaia DR3 (2020yCat.1350....0G)"),
    Target("Epsilon Indi B (Ba/Bb)", "eps Ind B",
           331.0441465885, -56.7828441951, 3981.977, -2466.832, 2000.0,
           "T-dwarf binary (Ba/Bb unresolved by SPHEREx, ~6\" pixels).",
           "SIMBAD / Gaia DR3 (2020yCat.1350....0G)"),
    Target("WISE 0855-0714", "WISE J085510.83-071442.5",
           133.81890825, -7.247048, -8123.7, 673.2, 2000.0,
           "Coldest known brown dwarf. The challenge-brief coordinate "
           "(08h55m10.831s -07d14m42.53s) is its ~2010.5 WISE-epoch position.",
           "SIMBAD (2021ApJS..253....7K)"),
    Target("RW Aurigae", "RW Aur",
           76.9565260346, 30.4014381739, 7.23, -20.33, 2000.0,
           "Young variable binary (T Tauri), known for deep dimming events.",
           "SIMBAD (2007A&A...474..653V)"),
    Target("R Draconis", "R Dra",
           248.1676108330, 66.7549794172, -20.643, 6.149, 2000.0,
           "Mira-type long-period variable (P ~ 246 d).",
           "SIMBAD / Gaia DR3 (2020yCat.1350....0G)"),
)

# The challenge brief's literal coordinate for WISE 0855, kept for reference.
WISE0855_BRIEF_COORD = ("08h55m10.831s", "-7d14m42.53s")


def target_by_name(name: str) -> Target | None:
    return next((t for t in CHALLENGE_TARGETS if t.name == name), None)
