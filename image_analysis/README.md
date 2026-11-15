# Image Analysis - SPHEREx Change Detection

Image Analysis Lead module for the NASA Space Apps 2026 *Planet X and SPHEREx* challenge.
It compares **real** SPHEREx observations of the same sky region at two times and finds
pixels / sources that moved or changed brightness.

Nothing in the app is simulated. If IRSA cannot be reached, the app says so and shows no data.

## Run (VS Code terminal, from the repository root)

Everything runs locally. No web server: results are written to a folder and the
HTML report opens straight from disk in your browser.

```bash
pip install -r image_analysis/requirements.txt

# recommended: spectrally matched pair (about 2-5 minutes, mostly IRSA download time)
python run_analysis.py --target r-dra --band D3 --match

# list observations with row numbers, then choose a pair yourself
python run_analysis.py --target r-dra --band D3 --list
python run_analysis.py --target r-dra --band D3 --earlier 5 --later 40

# other targets: barnard, eps-ind-b, wise0855, rw-aur, r-dra  - or any position:
python run_analysis.py --ra 133.8189 --dec -7.2470 --band D2
python run_analysis.py --help                # all options

python -m pytest image_analysis/tests -q     # unit tests (synthetic arrays only)
```

Each run creates `image_analysis_output/<target>_<band>_<time>/` (git-ignored) containing:

| File | Content |
|---|---|
| `report.html` | everything below in one page (opens automatically) |
| `1_comparison.png` | earlier vs later image, same grid and brightness stretch |
| `2_blink.gif` | blink comparison (animated) |
| `3_difference.png` | Later - Earlier heatmap (red brighter, blue fainter) |
| `4_significance.png` | noise-aware significance map |
| `5_candidate_stamps.png` | close-ups of each candidate |
| `candidates.csv`, `observations.csv`, `metadata.json` | tables + exact FITS source URLs |

## Data source

| Item | Value |
|---|---|
| Archive | NASA/IPAC Infrared Science Archive (IRSA) |
| Product | SPHEREx Level-2 spectral images (MEF FITS: IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE) |
| Collections | `spherex_dr1` (default, Data Release 1), `spherex_qr3` (Quick Release 3) - verified with `Irsa.list_collections(servicetype="SIA", filter="spherex")` on 2026-11-15, which also lists `spherex_dr1_deep`, `spherex_qr3_deep`, `spherex_dr1_cube` (not used here) |
| Query | `astroquery.ipac.irsa.Irsa.query_sia(pos=(coord, radius), collection=...)`; fallback: IRSA TAP `ivoa.obscore` |
| Download | `access_url` (full 2040x2040 frame, ~65 MB) or IRSA cutout `access_url?center=<ra>,<dec>deg&size=<n>arcmin` (all extensions kept) |
| Cache | astropy download cache (no hard-coded local paths) |

## Workflow

1. **Target** - five challenge targets (SIMBAD ICRS J2000.0 positions + proper motions, see `config.py`) or custom RA/Dec. Positions are propagated to the chosen date (`targets.position_at`).
2. **Search** - SIA2 query, one row per detector image (bands SPHEREx-D1..D6) with obs_id, UTC time, band/detector, wavelength range, `access_url`.
3. **Spectral matching** - SPHEREx uses linear variable filters, so the wavelength at the target depends on where it falls on the detector. Measured on real R Dra data, two D3 images saw the star at 1.64 vs 2.27 um. The *Measure* button reads each image's `WCS-WAVE` table at the target from 1-arcmin cutouts and suggests same-wavelength pairs with the longest baseline.
4. **Read FITS** (`fits_processor.py`) - IMAGE minus ZODI model; FLAGS bits named in `MP_*` header keywords (hard failures only) plus non-positive VARIANCE become NaN. Never median-imputed.
5. **Align** (`image_alignment.py`) - later epoch reprojected onto earlier WCS (incl. SIP) with **bicubic** interpolation (bilinear left ~30 sigma residuals on undersampled stars in tests; bicubic ~4). Bad-pixel mask reprojected separately and dilated. A residual FFT cross-correlation offset is reported.
6. **Subtract** (`image_subtraction.py`) - optional sigma-clipped sky removal, optional flux-scale fit, optional Gaussian PSF matching with correct variance propagation (`k^2` kernel). `Difference = Later - Earlier`.
7. **Significance** - `(L - E) / sqrt(Var_E + Var_L)` from VARIANCE extensions ("measured"); otherwise MAD-based uniform noise ("estimated", flagged as approximate). Optional **empirical noise rescale**: on real pairs the observed noise was 2-3x the VARIANCE prediction, so significances are divided by their robust width outside candidate regions; the factor is always shown.
8. **Candidates** (`candidate_detection.py`) - 8-connected regions above a configurable threshold, brightening/fading separately; centroid, RA/Dec, size, peak sigma, edge proximity, and **dipole** pairing (motion *or* misalignment). No object classification.

## Files

```
run_analysis.py                  command-line entry point (repo root)
image_analysis/
  config.py                      collections, thresholds, challenge targets
  targets.py                     coordinate parsing, proper-motion propagation
  nasa_data.py                   IRSA search / download / spectral matching
  fits_processor.py              FITS -> Observation (IMAGE/VARIANCE/FLAGS/ZODI/WCS-WAVE)
  image_alignment.py             WCS reprojection, residual-offset diagnostic
  image_subtraction.py           normalisation, difference, significance
  candidate_detection.py         grouping, coordinates, dipoles
  report.py                      PNG figures, blink GIF, self-contained HTML report
  cli.py                         argument parsing and the end-to-end run
  pipeline.py                    UI-independent end-to-end function
  utils.py                       pure-NumPy sigma clipping / MAD / ZScale
  tests/test_image_analysis.py   21 unit tests
```

## Using it from other code / the team dashboard

```python
from image_analysis import nasa_data
from image_analysis.pipeline import run_change_detection
df, service = nasa_data.search_observations(coord, radius_arcmin=0.5, bands=["SPHEREx-D3"])
a = nasa_data.fetch_observation(df.iloc[i], coord, size_arcmin=10)
b = nasa_data.fetch_observation(df.iloc[j], coord, size_arcmin=10)
result = run_change_detection(a, b, threshold=5.0)
result.diff.difference, result.diff.significance, result.candidates
```

The processing package has no UI dependency, so a notebook, script or a future team
dashboard can import `pipeline.run_change_detection` directly and use `report.build_report`
to produce the figures.

## Known limitations

- IRSA latency: each cutout takes ~15 s; SIA/TAP returned HTTP 502/503 intermittently during development (handled with retries + clear messages).
- Wavelength mismatch within a band produces real "changes" that are spectral, not temporal - use spectral matching.
- Reprojected variance is interpolated (noise correlation not modelled); hence the empirical rescale.
- Very bright stars (e.g. R Dra) are saturated/flagged at the core; changes appear in their wings and are marked "near masked pixels".
- Interpolating the later image around saturated star cores leaves black/white ringing, which can produce false candidates. These are marked "near masked pixels/edge" and should be treated as artefacts until checked by eye. One bright star at the cutout edge can also split into several candidates.
- SIMBAD proper motions are linear; fine to << 1 SPHEREx pixel over decades.
- On Windows machines with Application Control, `astropy.stats` can be blocked; this module therefore uses its own `utils.py` instead.
