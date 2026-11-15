"""Command-line runner: search IRSA, analyse two epochs, write a local report.

Examples (from the repository root):
    python run_analysis.py --target r-dra --list
    python run_analysis.py --target r-dra --band D3
    python run_analysis.py --target r-dra --band D3 --match
    python run_analysis.py --target r-dra --band D3 --earlier 0 --later 40
    python run_analysis.py --ra 133.79 --dec -7.25 --band D1
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

from . import config as cfg
from . import nasa_data
from .fits_processor import FitsReadError
from .pipeline import run_change_detection
from .targets import custom_target, parse_coordinates, position_at

ALIASES = {
    "barnard": "Barnard's Star", "eps-ind-b": "Epsilon Indi B (Ba/Bb)",
    "wise0855": "WISE 0855-0714", "rw-aur": "RW Aurigae", "r-dra": "R Draconis",
}


def _say(msg: str) -> None:
    print(msg, flush=True)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_analysis.py",
        description="SPHEREx change detection on real NASA IRSA data. Writes PNG images, "
                    "a blink GIF, CSV files and a self-contained HTML report to a local folder.")
    g = p.add_argument_group("target")
    g.add_argument("--target", choices=sorted(ALIASES), help="challenge target")
    g.add_argument("--ra", help="RA in degrees or hms (instead of --target)")
    g.add_argument("--dec", help="Dec in degrees or dms (instead of --target)")
    g.add_argument("--date", default="2025-10-01",
                   help="date the target position is propagated to for the search (default 2025-10-01)")
    a = p.add_argument_group("archive")
    a.add_argument("--collection", default=cfg.DEFAULT_COLLECTION, choices=cfg.SPHEREX_COLLECTIONS)
    a.add_argument("--band", help="D1..D6 (default: the band with the most observations)")
    a.add_argument("--fov", type=float, default=cfg.DEFAULT_CUTOUT_SIZE_ARCMIN,
                   help="cutout size in arcmin (default 10)")
    a.add_argument("--full-frame", action="store_true", help="download full 2040x2040 frames (~65 MB each)")
    s = p.add_argument_group("pair selection")
    s.add_argument("--list", action="store_true", help="only list the available observations and exit")
    s.add_argument("--earlier", type=int, help="row number from --list for the first epoch")
    s.add_argument("--later", type=int, help="row number from --list for the second epoch")
    s.add_argument("--match", action="store_true",
                   help="measure the wavelength at the target in every image (a few minutes) and use "
                        "the longest-baseline pair observed at the same wavelength (recommended)")
    s.add_argument("--max-mismatch", type=float, default=1.0,
                   help="max wavelength mismatch in percent for --match (default 1.0)")
    n = p.add_argument_group("analysis")
    n.add_argument("--threshold", type=float, default=cfg.DEFAULT_DETECTION_THRESHOLD_SIGMA)
    n.add_argument("--min-pixels", type=int, default=cfg.DEFAULT_MIN_CANDIDATE_PIXELS)
    n.add_argument("--psf-sigma", type=float, default=0.0, help="Gaussian PSF matching, pixels")
    n.add_argument("--match-flux", action="store_true", help="fit a photometric scale factor")
    n.add_argument("--no-zodi", action="store_true", help="do not subtract the ZODI model")
    n.add_argument("--no-rescale", action="store_true", help="disable empirical noise rescaling")
    n.add_argument("--blink-ms", type=int, default=700, help="blink GIF frame time (ms)")
    o = p.add_argument_group("output")
    o.add_argument("--out", help="output folder (default: image_analysis_output/<target>_<time>)")
    o.add_argument("--no-open", action="store_true", help="do not open the report automatically")
    args = p.parse_args(argv)
    if not args.target and not (args.ra and args.dec):
        p.error("give --target or both --ra and --dec")
    if (args.earlier is None) != (args.later is None):
        p.error("--earlier and --later must be used together")
    return args


def _list(df: pd.DataFrame) -> None:
    cols = [c for c in ("obs_id", "obs_date_utc", "energy_bandpassname", "em_min_um",
                        "em_max_um", "lambda_target_um") if c in df]
    with pd.option_context("display.max_rows", 500, "display.width", 160,
                           "display.float_format", "{:.3f}".format):
        print(df[cols].to_string())


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    # SPHEREx headers trigger an astropy INFO notice about SIP + TAN CTYPEs (SIP is
    # correctly applied); silence it for the whole run, including worker threads.
    from astropy import log
    log.setLevel("WARNING")

    # ---- target -----------------------------------------------------------
    if args.target:
        target = cfg.target_by_name(ALIASES[args.target])
        coord = position_at(target, Time(args.date))
        name = args.target
    else:
        try:
            c0 = parse_coordinates(args.ra, args.dec)
        except Exception:  # noqa: BLE001
            _say("ERROR: could not parse --ra/--dec"); return 2
        target, coord, name = custom_target(c0.ra.deg, c0.dec.deg), c0, "custom"
    _say(f"Target   {target.name}   RA {coord.ra.deg:.6f}  Dec {coord.dec.deg:+.6f}  ({args.date})")

    # ---- search -----------------------------------------------------------
    band = f"SPHEREx-{args.band.upper()}" if args.band and not args.band.upper().startswith("SPHEREX") else args.band
    _say(f"Searching NASA IRSA ({args.collection})...")
    try:
        df, service = nasa_data.search_observations(coord, 0.5, args.collection,
                                                    [band] if band else None)
    except nasa_data.ArchiveError as exc:
        _say(f"ERROR: {exc}"); return 1
    if df.empty:
        _say("IRSA has no SPHEREx images covering this position for that band/collection."); return 1
    if not band:
        band = df["energy_bandpassname"].value_counts().idxmax()
        df = df[df["energy_bandpassname"] == band].reset_index(drop=True)
    _say(f"Found    {len(df)} {band} images via {service}, "
         f"{str(df.obs_date_utc.iloc[0])[:10]} .. {str(df.obs_date_utc.iloc[-1])[:10]}")
    if len(df) < 2:
        _say("Only one observation - at least two epochs are needed."); return 1

    if args.match:
        _say(f"Measuring wavelength at the target in {len(df)} images (about 15 s each, 4 at a time)...")
        df = nasa_data.measure_wavelengths(
            df, coord, progress=lambda f: print(f"\r  {f:5.0%}", end="", flush=True))
        print()
    if args.list:
        _list(df); return 0

    # ---- pair -------------------------------------------------------------
    warnings: list[str] = []
    if args.earlier is not None:
        try:
            ra_, rb_ = df.iloc[args.earlier], df.iloc[args.later]
        except IndexError:
            _say(f"ERROR: row numbers must be between 0 and {len(df) - 1} (see --list)"); return 2
    elif args.match:
        pairs = nasa_data.suggest_pairs(df, max_rel_dlambda=args.max_mismatch / 100)
        if pairs.empty:
            _say("No pair observed at matching wavelengths - try --max-mismatch 3"); return 1
        best = pairs.iloc[0]
        ra_, rb_ = df.loc[best.earlier_index], df.loc[best.later_index]
        _say(f"Matched  {best.earlier} vs {best.later}: {best.baseline_days:.0f} days, "
             f"{best.lambda_um:.3f} um, mismatch {best.dlambda_rel_pct:.2f}%")
    else:
        ra_, rb_ = df.iloc[0], df.iloc[-1]
        warnings.append("Pair chosen by longest time baseline without wavelength matching; "
                        "re-run with --match to remove spectral (colour) differences.")

    # Centre cutouts on the target position midway between the two epochs.
    mid = Time((float(ra_.t_min) + float(rb_.t_min)) / 2, format="mjd")
    centre = position_at(target, mid)
    size = None if args.full_frame else args.fov

    _say(f"Download {ra_.obs_id} ({str(ra_.obs_date_utc)[:19]})")
    _say(f"         {rb_.obs_id} ({str(rb_.obs_date_utc)[:19]})")
    try:
        a = nasa_data.fetch_observation(ra_, centre, size, subtract_zodi=not args.no_zodi)
        b = nasa_data.fetch_observation(rb_, centre, size, subtract_zodi=not args.no_zodi)
        _say("Analyse  align -> subtract -> significance -> candidates")
        res = run_change_detection(a, b, threshold=args.threshold, min_pixels=args.min_pixels,
                                   match_flux=args.match_flux, psf_sigma_pix=args.psf_sigma,
                                   remove_background=True, rescale_noise=not args.no_rescale)
    except (nasa_data.ArchiveError, FitsReadError, ValueError) as exc:
        _say(f"ERROR: {exc}"); return 1

    wl = (res.pair.earlier.wavelength_um, res.pair.later.wavelength_um)
    if all(wl) and abs(wl[0] - wl[1]) > 0.02 * wl[0]:
        warnings.append(f"Wavelength at the target differs: {wl[0]:.3f} vs {wl[1]:.3f} um - part of "
                        "the difference is spectral, not temporal.")

    # ---- report -----------------------------------------------------------
    target_pix = None
    if np.hypot(target.pm_ra_cosdec_mas_yr, target.pm_dec_mas_yr) >= 1:
        target_pix = {}
        for label, o in (("Target position @ earlier", res.pair.earlier),
                         ("Target position @ later", res.pair.later)):
            x, y = res.pair.earlier.wcs.world_to_pixel(position_at(target, float(o.meta["t_min"])))
            target_pix[label] = (float(x), float(y))

    e, l = res.pair.earlier, res.pair.later
    info = {
        "Target": target.name,
        "Position (ICRS)": SkyCoord(centre).to_string("hmsdms", precision=2),
        "Archive": f"NASA/IPAC IRSA, collection {args.collection}, via {service}",
        "Band": band,
        "Earlier": f"{e.meta.get('obs_id')}  {str(e.meta.get('obs_date_utc'))[:19]} UTC"
                   + (f"  |  {e.wavelength_um:.4f} um" if e.wavelength_um else ""),
        "Later": f"{l.meta.get('obs_id')}  {str(l.meta.get('obs_date_utc'))[:19]} UTC"
                 + (f"  |  {l.wavelength_um:.4f} um" if l.wavelength_um else ""),
        "Time baseline": f"{float(l.meta['t_min']) - float(e.meta['t_min']):.2f} days",
        "Image size": f"{e.shape[1]} x {e.shape[0]} pixels ({'full frame' if args.full_frame else f'{args.fov:g} arcmin cutout'})",
        "Significance": f"{res.diff.significance_kind} (VARIANCE)" if res.diff.significance_kind == "measured"
                        else "estimated (no VARIANCE)",
        "Noise rescale factor": f"{res.noise_scale:.2f}",
        "Threshold": f"{args.threshold:g} sigma, >= {args.min_pixels} pixels",
        "Candidates": str(len(res.candidates)),
        "Earlier FITS": e.meta.get("source_url", ""),
        "Later FITS": l.meta.get("source_url", ""),
        "_warnings": warnings,
    }
    out = Path(args.out) if args.out else Path("image_analysis_output") / \
        f"{name}_{band.split('-')[-1]}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    ignore = out.parent / ".gitignore"
    if out.parent.name == "image_analysis_output" and not ignore.exists():
        ignore.write_text("# generated reports - not committed\n*\n", encoding="utf-8")
    from .report import build_report  # imported late: matplotlib is slow to load
    _say("Writing  images and report...")
    files = build_report(res, out, info, target_pix, args.blink_ms)
    df.to_csv(out / "observations.csv", index=False)

    for w in warnings:
        _say(f"NOTE     {w}")
    _say(f"Result   {len(res.candidates)} candidate regions at >= {args.threshold:g} sigma "
         f"(noise rescale {res.noise_scale:.2f})")
    _say(f"Saved    {out.resolve()}")
    for p in sorted(out.iterdir()):
        _say(f"         - {p.name}")
    if not args.no_open:
        report = files["report"].resolve()
        if sys.platform == "win32":
            os.startfile(report)  # noqa: S606 - opens the local file in the default browser
        else:
            webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
