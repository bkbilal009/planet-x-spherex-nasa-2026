"""Unit tests for the image-analysis algorithms.

All arrays here are SYNTHETIC test fixtures used only to verify algorithm
behaviour. They are never presented as NASA observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from image_analysis import nasa_data
from image_analysis.candidate_detection import detect_candidates
from image_analysis.config import CHALLENGE_TARGETS, target_by_name
from image_analysis.fits_processor import FitsReadError, Observation, read_spherex_fits
from image_analysis.image_alignment import align_pair, estimate_residual_shift
from image_analysis.image_subtraction import compute_difference
from image_analysis.pipeline import run_change_detection
from image_analysis.targets import parse_coordinates, position_at

N = 64
SCALE_DEG = 6.15 / 3600
RA0, DEC0 = 133.8, -7.25


def make_wcs(crpix=(32.5, 32.5)) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = [-SCALE_DEG, SCALE_DEG]
    return w


def gaussian(x0, y0, amp=50.0, sigma=1.2, n=N):
    y, x = np.mgrid[0:n, 0:n]
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def star_field(wcs: WCS, sources) -> np.ndarray:
    """Render sky sources (ra, dec, amp) on the pixel grid of ``wcs``."""
    img = np.zeros((N, N))
    for ra, dec, amp in sources:
        x, y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        img += gaussian(float(x), float(y), amp)
    return img


SKY = [(RA0 + 0.01, DEC0 - 0.008, 80.0), (RA0 - 0.012, DEC0 + 0.01, 60.0),
       (RA0 + 0.004, DEC0 + 0.015, 40.0), (RA0 - 0.02, DEC0 - 0.02, 70.0)]


def make_obs(sources, crpix=(32.5, 32.5), var=0.04, seed=0, t=60800.0) -> Observation:
    rng = np.random.default_rng(seed)
    w = make_wcs(crpix)
    img = star_field(w, sources) + rng.normal(0, np.sqrt(var), (N, N))
    return Observation(image=img, wcs=w, variance=np.full((N, N), var),
                       bad_mask=np.zeros((N, N), bool), meta={"t_min": t})


def write_mef(path, image, variance=None, flags=None, zodi=None, flag_bits=None):
    hdr = make_wcs().to_header()
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(image, header=hdr, name="IMAGE")]
    if flags is not None:
        fh = fits.Header()
        for name, bit in (flag_bits or {}).items():
            fh[f"MP_{name}"] = bit
        hdus.append(fits.ImageHDU(flags, header=fh, name="FLAGS"))
    if variance is not None:
        hdus.append(fits.ImageHDU(variance, name="VARIANCE"))
    if zodi is not None:
        hdus.append(fits.ImageHDU(zodi, name="ZODI"))
    fits.HDUList(hdus).writeto(path)
    return path


# --------------------------------------------------------------------- FITS
def test_read_full_mef(tmp_path):
    img = np.ones((N, N)) * 5.0
    flags = np.zeros((N, N), np.int32)
    flags[3, 4] = 1 << 7          # HOT -> bad
    flags[5, 5] = 1 << 20         # unknown/non-fatal bit -> kept
    zodi = np.full((N, N), 2.0)
    p = write_mef(tmp_path / "a.fits", img, np.full((N, N), 0.1), flags, zodi,
                  {"HOT": 7, "SRC": 20})
    obs = read_spherex_fits(p, meta={"obs_id": "x"})
    assert obs.has_variance and obs.wcs.has_celestial
    assert np.isnan(obs.image[3, 4]) and obs.bad_mask[3, 4]
    assert not obs.bad_mask[5, 5]
    assert obs.image[10, 10] == pytest.approx(3.0)       # zodi subtracted
    assert obs.meta["zodi_subtracted"] is True


def test_read_missing_optional_extensions(tmp_path):
    p = write_mef(tmp_path / "b.fits", np.ones((N, N)))
    obs = read_spherex_fits(p)
    assert obs.variance is None and obs.zodi is None
    assert not obs.bad_mask.any()


def test_read_invalid_and_missing_files(tmp_path):
    bad = tmp_path / "bad.fits"
    bad.write_text("not a fits file")
    with pytest.raises(FitsReadError):
        read_spherex_fits(bad)
    with pytest.raises(FitsReadError):
        read_spherex_fits(tmp_path / "nope.fits")


def test_nonpositive_variance_marked_bad(tmp_path):
    var = np.full((N, N), 0.1)
    var[7, 7] = 0.0
    obs = read_spherex_fits(write_mef(tmp_path / "c.fits", np.ones((N, N)), var))
    assert obs.bad_mask[7, 7] and np.isnan(obs.image[7, 7])


# ---------------------------------------------------------------- alignment
def test_alignment_removes_static_stars():
    a = make_obs(SKY, seed=1, t=1.0)
    b = make_obs(SKY, crpix=(30.2, 35.7), seed=2, t=2.0)   # different pointing
    pair = align_pair(a, b)
    res = compute_difference(pair)
    core = pair.footprint.copy()
    core[:6], core[-6:], core[:, :6], core[:, -6:] = False, False, False, False
    # Static stars cancel to within a few noise sigma after reprojection.
    assert np.nanmax(np.abs(res.significance[core])) < 6.0
    dy, dx = pair.residual_shift_pix
    assert abs(dx) < 0.3 and abs(dy) < 0.3


def test_residual_shift_estimate():
    a = gaussian(30, 30) + gaussian(10, 45)
    b = gaussian(32, 29) + gaussian(12, 44)            # moved +2 x, -1 y
    dy, dx = estimate_residual_shift(a, b)
    assert dx == pytest.approx(2, abs=0.2) and dy == pytest.approx(-1, abs=0.2)


# -------------------------------------------------------------- subtraction
def test_difference_sign_and_nan_handling():
    a = make_obs(SKY, seed=3, t=1.0)
    b = make_obs(SKY, seed=4, t=2.0)
    b.image += gaussian(20, 20, amp=10.0)               # later brightening
    b.image[40:43, 40:43] = np.nan                      # invalid pixels
    b.bad_mask[40:43, 40:43] = True
    pair = align_pair(a, b)
    res = compute_difference(pair)
    assert res.difference[20, 20] > 5                   # positive = brighter later
    assert np.isnan(res.difference[41, 41])
    assert not res.valid[41, 41]
    assert res.significance_kind == "measured"


def test_significance_formula_and_zero_variance():
    a = make_obs([], var=1.0, seed=5, t=1.0)
    b = make_obs([], var=3.0, seed=6, t=2.0)
    a.image[:] = 0.0
    b.image[:] = 0.0
    b.image[10, 10] = 8.0
    a.variance[30, 30] = 0.0
    pair = align_pair(a, b)
    res = compute_difference(pair, remove_background=False)
    # (8 - 0) / sqrt(1 + 3) = 4 (reprojection onto an identical grid is exact)
    assert res.significance[10, 10] == pytest.approx(4.0, rel=1e-6)
    assert np.isnan(res.significance[30, 30]) and not res.valid[30, 30]


def test_significance_estimated_without_variance():
    a = make_obs(SKY, seed=7, t=1.0)
    b = make_obs(SKY, seed=8, t=2.0)
    a.variance = b.variance = None
    res = compute_difference(align_pair(a, b))
    assert res.significance_kind == "estimated"
    assert any("approximate" in n for n in res.notes)
    assert np.nanstd(res.significance) == pytest.approx(1.0, abs=0.2)


def test_flux_scale_matching():
    a = make_obs(SKY, seed=9, t=1.0)
    b = make_obs(SKY, seed=10, t=2.0)
    b.image *= 1.1
    b.variance *= 1.21
    res = compute_difference(align_pair(a, b), match_flux=True)
    assert res.flux_scale == pytest.approx(1.1, rel=0.03)


def test_psf_smoothing_keeps_significance_calibrated():
    a = make_obs([], seed=11, t=1.0)
    b = make_obs([], seed=12, t=2.0)
    res = compute_difference(align_pair(a, b), psf_sigma_pix=1.5)
    core = res.significance[8:-8, 8:-8]
    assert np.nanstd(core) == pytest.approx(1.0, abs=0.25)


def test_empirical_noise_rescale():
    # Variance under-stated 3x in std: rescale should recover ~3 and leave
    # calibrated maps alone.
    a = make_obs(SKY, seed=17, t=1.0)
    b = make_obs(SKY, seed=18, t=2.0)
    a.variance = a.variance / 9
    b.variance = b.variance / 9
    r = run_change_detection(a, b, threshold=5.0)
    assert r.noise_scale == pytest.approx(3.0, rel=0.15)
    assert r.candidates.empty
    r_ok = run_change_detection(make_obs(SKY, seed=19, t=1.0), make_obs(SKY, seed=20, t=2.0))
    assert r_ok.noise_scale < 1.2


# ---------------------------------------------------------------- candidates
def test_moving_source_detected_as_dipole():
    mover_a = (RA0 + 0.0, DEC0 + 0.0, 30.0)
    mover_b = (RA0 - 6 * SCALE_DEG, DEC0 + 0.0, 30.0)   # moved 6 px
    a = make_obs(SKY + [mover_a], seed=13, t=1.0)
    b = make_obs(SKY + [mover_b], seed=14, t=2.0)
    r = run_change_detection(b, a, threshold=5.0)       # order is fixed by t_min
    c = r.candidates
    assert {1, -1} <= set(c.sign)
    pos = c[c.sign > 0].iloc[0]
    neg = c[c.sign < 0].iloc[0]
    # Fading at old position, brightening at new position (RA increases to the left).
    old = SkyCoord(mover_a[0] * u.deg, mover_a[1] * u.deg)
    new = SkyCoord(mover_b[0] * u.deg, mover_b[1] * u.deg)
    assert SkyCoord(neg.ra_deg * u.deg, neg.dec_deg * u.deg).separation(old) < 4 * u.arcsec
    assert SkyCoord(pos.ra_deg * u.deg, pos.dec_deg * u.deg).separation(new) < 4 * u.arcsec


def test_no_candidates_in_pure_noise_and_threshold_config():
    a = make_obs(SKY, seed=15, t=1.0)
    b = make_obs(SKY, seed=16, t=2.0)
    r = run_change_detection(a, b, threshold=6.0)
    assert r.candidates.empty
    assert list(r.candidates.columns)                   # schema still present


def test_candidates_ignore_invalid_pixels():
    sig = np.zeros((N, N))
    sig[10:12, 10:12] = 20
    valid = np.ones((N, N), bool)
    valid[10:12, 10:12] = False
    assert detect_candidates(sig, sig, valid).empty


# ----------------------------------------------------------- archive access
def test_archive_failure_raises_clean_error(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("502 Proxy Error")
    monkeypatch.setattr(nasa_data, "_query_sia", boom)
    monkeypatch.setattr(nasa_data, "_query_tap", boom)
    monkeypatch.setattr(nasa_data.time, "sleep", lambda s: None)
    with pytest.raises(nasa_data.ArchiveError, match="temporarily unavailable"):
        nasa_data.search_observations(SkyCoord(RA0 * u.deg, DEC0 * u.deg))


def test_archive_empty_result_is_not_error(monkeypatch):
    monkeypatch.setattr(nasa_data, "_query_sia", lambda *a: pd.DataFrame(
        columns=["obs_id", "t_min", "energy_bandpassname", "access_url"]))
    df, service = nasa_data.search_observations(SkyCoord(RA0 * u.deg, DEC0 * u.deg))
    assert df.empty and service == "SIA2"


def test_tap_fallback_and_tidy(monkeypatch):
    monkeypatch.setattr(nasa_data, "_query_sia",
                        lambda *a: (_ for _ in ()).throw(TimeoutError("down")))
    monkeypatch.setattr(nasa_data.time, "sleep", lambda s: None)
    monkeypatch.setattr(nasa_data, "_query_tap", lambda *a: pd.DataFrame({
        "obs_id": ["b", "a", "a"], "t_min": [61000.0, 60800.0, 60800.0],
        "energy_bandpassname": ["SPHEREx-D2", "SPHEREx-D2", "SPHEREx-D2"],
        "em_min": [1.1e-6] * 3, "em_max": [1.6e-6] * 3, "access_url": ["u2", "u1", "u1"]}))
    df, service = nasa_data.search_observations(SkyCoord(RA0 * u.deg, DEC0 * u.deg),
                                                bands=["SPHEREx-D2"])
    assert service.startswith("TAP")
    assert list(df.obs_id) == ["a", "b"]                # deduplicated, time-sorted
    assert df.detector.iloc[0] == "2" and df.em_min_um.iloc[0] == pytest.approx(1.1)


def test_cutout_url_format():
    url = nasa_data.cutout_url("https://x/y.fits", SkyCoord(10 * u.deg, -5 * u.deg), 6)
    assert url == "https://x/y.fits?center=10.0000000,-5.0000000deg&size=6.0000arcmin"


# ------------------------------------------------------------------ targets
def test_wise0855_propagates_to_brief_coordinate():
    t = target_by_name("WISE 0855-0714")
    brief = SkyCoord("08h55m10.831s", "-7d14m42.53s")
    # The brief coordinate is the ~2010.45 (WISE-epoch) position.
    pos = position_at(t, Time(2010.45, format="jyear"))
    assert pos.separation(brief) < 1.5 * u.arcsec


def test_all_targets_defined_and_parse():
    assert len(CHALLENGE_TARGETS) == 5
    c = parse_coordinates("08h55m10.831s", "-7d14m42.53s")
    assert c.ra.deg == pytest.approx(133.795129, abs=1e-5)
    assert parse_coordinates("10.5", "-3").dec.deg == -3
