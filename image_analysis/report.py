"""Static, local outputs: high-resolution PNG figures, blink GIF, HTML report.

Everything is written to a folder on disk and viewed offline - no web server.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .candidate_detection import candidate_cutout  # noqa: E402
from .utils import symmetric_limit, zscale_limits  # noqa: E402
from .pipeline import ChangeDetectionResult  # noqa: E402

BG, PANEL, TEXT, MUTED = "#05070d", "#0c111c", "#d7deea", "#8b95a7"
DIVERGING = "RdBu_r"   # blue = fainter, white = no change, red = brighter
DPI = 160

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#243049", "font.size": 10, "axes.titlesize": 11,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
})


def common_limits(earlier: np.ndarray, later: np.ndarray) -> tuple[float, float]:
    """One ZScale stretch from both epochs, so only real changes look different."""
    both = np.concatenate([earlier[np.isfinite(earlier)], later[np.isfinite(later)]])
    return zscale_limits(both)


def _gray():
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#1a1f2b")
    return cmap


def _div():
    cmap = plt.get_cmap(DIVERGING).copy()
    cmap.set_bad("#1a1f2b")
    return cmap


def _label(o) -> str:
    m = o.meta
    wl = f"  |  {o.wavelength_um:.3f} um" if o.wavelength_um else ""
    return f"{m.get('obs_id', m.get('file', ''))}\n{str(m.get('obs_date_utc', ''))[:19]} UTC{wl}"


def _target_marks(ax, res: ChangeDetectionResult, target_pix: dict | None) -> None:
    if not target_pix:
        return
    for name, (x, y), color in ((k, v, c) for (k, v), c in
                                zip(target_pix.items(), ("#7fd4ff", "#ffb347"))):
        ax.plot(x, y, marker="x", ms=11, mew=2, color=color, label=name, ls="none")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False,
              fontsize=8, labelcolor=TEXT)


def _cand_marks(ax, cands: pd.DataFrame) -> None:
    for r in cands.itertuples():
        color = "#ffb347" if r.sign > 0 else "#7fd4ff"
        ax.add_patch(plt.Circle((r.x, r.y), 3.5, fill=False, lw=1.4, color=color))
        ax.text(r.x + 4, r.y + 4, f"#{r.id}", color=color, fontsize=8, weight="bold")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_comparison(res, path: Path, target_pix=None) -> Path:
    lim = common_limits(res.diff.earlier, res.diff.later)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
    for ax, arr, role, o in ((axes[0], res.diff.earlier, "EARLIER", res.pair.earlier),
                             (axes[1], res.diff.later, "LATER (aligned)", res.pair.later)):
        im = ax.imshow(arr, origin="lower", cmap=_gray(), vmin=lim[0], vmax=lim[1],
                       interpolation="nearest")
        ax.set_title(f"{role}\n{_label(o)}", loc="left")
        ax.set_xlabel("x [pixel]"); ax.set_ylabel("y [pixel]")
        _target_marks(ax, res, target_pix)
    fig.colorbar(im, ax=axes, shrink=0.85,
                 label="Surface brightness [MJy/sr]  (same grid, same stretch)")
    return _save(fig, path)


def fig_difference(res, path: Path, target_pix=None) -> Path:
    d = res.diff.difference
    lim = symmetric_limit(d)
    fig, ax = plt.subplots(figsize=(8.6, 7.4), constrained_layout=True)
    im = ax.imshow(d, origin="lower", cmap=_div(), norm=TwoSlopeNorm(0, -lim, lim),
                   interpolation="nearest")
    ax.set_title("Difference = Later - Earlier", loc="left")
    ax.set_xlabel("x [pixel]"); ax.set_ylabel("y [pixel]")
    _cand_marks(ax, res.candidates)
    _target_marks(ax, res, target_pix)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Change [MJy/sr]   (red = brighter later, blue = fainter later)")
    return _save(fig, path)


def fig_significance(res, path: Path, target_pix=None) -> Path:
    s = res.diff.significance
    lim = max(2 * res.threshold, 3.0)
    fig, ax = plt.subplots(figsize=(8.6, 7.4), constrained_layout=True)
    im = ax.imshow(s, origin="lower", cmap=_div(), norm=TwoSlopeNorm(0, -lim, lim),
                   interpolation="nearest")
    kind = res.diff.significance_kind
    extra = f", rescaled /{res.noise_scale:.2f}" if res.noise_scale > 1.05 else ""
    ax.set_title(f"Significance [sigma]  ({kind}{extra})", loc="left")
    ax.set_xlabel("x [pixel]"); ax.set_ylabel("y [pixel]")
    _cand_marks(ax, res.candidates)
    _target_marks(ax, res, target_pix)
    fig.colorbar(im, ax=ax, shrink=0.9, label=f"(L - E) / noise   threshold = +/-{res.threshold:g} sigma")
    return _save(fig, path)


def fig_candidates(res, path: Path, max_rows: int = 8) -> Path | None:
    c = res.candidates.head(max_rows)
    if c.empty:
        return None
    cols = (("Earlier", res.diff.earlier, False), ("Later", res.diff.later, False),
            ("Difference", res.diff.difference, True), ("Significance", res.diff.significance, True))
    fig, axes = plt.subplots(len(c), 4, figsize=(9, 2.3 * len(c)), squeeze=False,
                             constrained_layout=True)
    for i, r in enumerate(c.itertuples()):
        # Earlier and Later stamps share one stretch so only real changes differ.
        shared = common_limits(candidate_cutout(res.diff.earlier, r.x, r.y, 12),
                               candidate_cutout(res.diff.later, r.x, r.y, 12))
        for j, (name, arr, div) in enumerate(cols):
            st = candidate_cutout(arr, r.x, r.y, 12)
            ax = axes[i, j]
            if div:
                lim = symmetric_limit(st, 100)
                ax.imshow(st, origin="lower", cmap=_div(), vmin=-lim, vmax=lim, interpolation="nearest")
            else:
                lo, hi = shared
                ax.imshow(st, origin="lower", cmap=_gray(), vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name)
        axes[i, 0].set_ylabel(f"#{r.id}  {r.peak_sigma:+.1f} sigma", color=TEXT, fontsize=9)
    return _save(fig, path)


def blink_gif(res, path: Path, interval_ms: int = 700, size_px: int = 640) -> Path:
    """Animated GIF alternating the two aligned epochs with a shared stretch."""
    from PIL import Image, ImageDraw

    lim = common_limits(res.diff.earlier, res.diff.later)
    frames = []
    for arr, label in ((res.diff.earlier, "EARLIER  " + str(res.pair.earlier.meta.get("obs_date_utc", ""))[:19]),
                       (res.diff.later, "LATER    " + str(res.pair.later.meta.get("obs_date_utc", ""))[:19])):
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1)); ax.set_axis_off()
        ax.imshow(arr, origin="lower", cmap=_gray(), vmin=lim[0], vmax=lim[1],
                  interpolation="nearest", aspect="equal")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
        img = Image.open(buf).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle((8, 8, 8 + 7 * len(label), 30), fill=(5, 7, 13))
        draw.text((14, 13), label, fill=(215, 222, 234))
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=interval_ms, loop=0)
    return path


def _img_tag(path: Path | None, alt: str) -> str:
    if path is None:
        return ""
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{data}" alt="{html.escape(alt)}">'


def write_html(res, out: Path, files: dict[str, Path | None], info: dict) -> Path:
    """Single self-contained HTML file (images embedded) - open by double-click."""
    c = res.candidates
    rows = "".join(
        f"<tr><td>#{r.id}</td><td>{'brighter' if r.sign > 0 else 'fainter'}</td>"
        f"<td>{r.peak_sigma:+.1f}</td><td>{r.n_pix}</td><td>{r.x:.1f}, {r.y:.1f}</td>"
        f"<td>{'' if np.isnan(r.ra_deg) else f'{r.ra_deg:.6f}, {r.dec_deg:+.6f}'}</td>"
        f"<td>{html.escape(r.notes)}</td></tr>" for r in c.itertuples())
    cand_table = (f"<table><tr><th>ID</th><th>Change</th><th>Peak sigma</th><th>Pixels</th>"
                  f"<th>x, y</th><th>RA, Dec [deg]</th><th>Notes</th></tr>{rows}</table>"
                  if not c.empty else "<p class='muted'>No regions exceed the threshold.</p>")
    meta_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in info.items() if not k.startswith("_"))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in res.diff.notes + info.get("_warnings", []))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPHEREx Change Report</title><style>
:root{{--bg:{BG};--panel:{PANEL};--line:#1f2a40;--text:{TEXT};--muted:{MUTED}}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 "Segoe UI",Inter,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 60px}}
h1{{margin:0 0 4px;font-size:26px}} h2{{margin:36px 0 10px;font-size:18px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.muted{{color:var(--muted)}} img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#000}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--panel);border-radius:8px;overflow:hidden}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:500}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media (max-width:800px){{.grid{{grid-template-columns:1fr}}}} li{{margin:3px 0}}
.box{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px}}
</style></head><body><main>
<h1>SPHEREx Change Detection Report</h1>
<p class="muted">Real NASA/IPAC IRSA SPHEREx observations. Candidates are pixel-level changes for human
inspection, not object classifications.</p>
<h2>Observations</h2><table>{meta_rows}</table>
<h2>1. Side-by-side comparison</h2>{_img_tag(files.get("comparison"), "comparison")}
<h2>2. Blink comparison</h2><p class="muted">Moving sources jump, variable sources pulse, static stars stay still.</p>
{_img_tag(files.get("blink"), "blink")}
<h2>3. Difference heatmap</h2><p class="muted">Red = brighter in the later image, blue = fainter, white = no change.
A red/blue pair can mean motion - or residual misalignment.</p>
{_img_tag(files.get("difference"), "difference")}
<h2>4. Significance map</h2><p class="muted">(Later - Earlier) / sqrt(Var_earlier + Var_later).</p>
{_img_tag(files.get("significance"), "significance")}
<h2>5. Candidate changes ({len(c)})</h2>{cand_table}
<p></p>{_img_tag(files.get("candidates"), "candidate stamps")}
<h2>Processing notes</h2><div class="box"><ul>{notes or "<li>None</li>"}</ul></div>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_report(res: ChangeDetectionResult, out: Path, info: dict,
                 target_pix: dict | None = None, blink_ms: int = 700) -> dict[str, Path | None]:
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "comparison": fig_comparison(res, out / "1_comparison.png", target_pix),
        "blink": blink_gif(res, out / "2_blink.gif", blink_ms),
        "difference": fig_difference(res, out / "3_difference.png", target_pix),
        "significance": fig_significance(res, out / "4_significance.png", target_pix),
        "candidates": fig_candidates(res, out / "5_candidate_stamps.png"),
    }
    res.candidates.to_csv(out / "candidates.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(
        {k: v for k, v in info.items() if not k.startswith("_")}, indent=2, default=str),
        encoding="utf-8")
    files["report"] = write_html(res, out, files, info)
    return files
