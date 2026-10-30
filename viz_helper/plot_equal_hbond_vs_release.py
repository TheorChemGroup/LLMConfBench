#!/usr/bin/env python3

"""CRA on equal-H-bond pairs vs release date.

Equal-H-bond pairs carry the same number of intramolecular H-bonds on both
sides (0-vs-0 and 1-vs-1), so the H-bond heuristic gives no signal: 2,997 of
the 3,802 reliable pairs (|ΔE| >= 2.0 kcal/mol, r2SCAN-3c). Mean per-molecule
Kendall tau pooled over three seeds; force fields as squares.

    python viz_helper/plot_equal_hbond_vs_release.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoy_common import (
    BUNDLES,
    bench_molecules,
    equal_hbond_pair_tables,
    hb_count_map,
    load_r2scan,
    override_r2scan3c,
    score_pairs_ff,
    score_pairs_letters,
)
from plot_metrics_vs_release import (
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
)
from plot_tau_vs_release_square import (
    _date_to_x,
    _jitter_xs,
    _set_date_axis,
    open_square_figure,
    save_square_figure,
)

from config.models_config import (
    family_for_display,
    family_legend_color,
    marker_for_display,
    model_files_by_display,
    release_dates_by_display,
)
from viz.plot_style import (
    REF_GFNFF,
    REF_MMFF,
    REF_MMFF_LINESTYLE,
    REF_UFF,
    style_ax,
)

REPO = Path(__file__).resolve().parent.parent
GEOM = REPO / "geom-qm9"
OUT = Path("figures/cra_fair_vs_release.png")
YLIM = (-0.1, 1.0)
YTICKS = np.arange(-0.1, 1.01, 0.1)


def _macro(stats: dict) -> tuple[float, float, int]:
    accs = np.asarray([v[0] / v[1] for v in stats.values() if v[1]])
    taus = 2 * accs - 1
    n = len(taus)
    if not n:
        return float("nan"), 0.0, 0
    half = (float(np.std(taus, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    if n > 1:
        from scipy.stats import t as t_dist
        half *= float(t_dist.ppf(0.975, n - 1))
    return float(np.mean(taus)), half, n


def collect_and_write(out: Path | None = None) -> Path:
    out = Path(out) if out is not None else OUT
    r2scan = load_r2scan()
    bench = bench_molecules()
    counts = hb_count_map(r2scan, bench)
    table = equal_hbond_pair_tables(r2scan, counts)
    n_pairs = sum(len(v) for v in table.values())
    override = override_r2scan3c()
    bdirs = [GEOM / b for b in BUNDLES]

    pts = []
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            continue
        st = score_pairs_letters(bdirs, pred, {"eq": table}, override)["eq"]
        tau, half, _ = _macro(st)
        pts.append((display, release_dates_by_display()[display], tau, half))
    ff_pts = []
    for name, getter in [("UFF", get_uff_ranking), ("GFN-FF", get_gfnff_ranking),
                         ("MMFF94", get_mmff94_ranking)]:
        st = score_pairs_ff(getter, {"eq": table}, r2scan)["eq"]
        tau, _, _ = _macro(st)
        ff_pts.append((name, tau))
    print(f"equal-H-bond pairs: {n_pairs}, models: {len(pts)}, "
          f"FF: {[(n, f'{t:+.3f}') for n, t in ff_pts]}", flush=True)

    present = [p[0] for p in pts]
    xs = _jitter_xs([_date_to_x(p[1]) for p in pts])
    fig, ax, _legend_ax = open_square_figure()
    for (display, _, tau, half), x in zip(pts, xs):
        fam = family_for_display(display)
        ax.errorbar([x], [tau], yerr=half, fmt="none", ecolor="#BDBDBD",
                    elinewidth=0.8, capsize=0, zorder=2)
        ax.scatter([x], [tau], s=28, c=family_legend_color(fam, present),
                   marker=marker_for_display(display), zorder=3)
    for name, tau in ff_pts:
        col = {"UFF": REF_UFF, "GFN-FF": REF_GFNFF, "MMFF94": REF_MMFF}[name]
        ls = {"UFF": ":", "GFN-FF": "-.", "MMFF94": REF_MMFF_LINESTYLE}[name]
        if np.isfinite(tau):
            ax.axhline(tau, color=col, ls=ls, lw=1.3, zorder=1)
    ax.axhline(0.0, color="grey", ls=":", lw=0.8, zorder=1)
    _set_date_axis(ax)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    style_ax(ax, square=False)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("CRA across fair pairs", fontsize=10)
    save_square_figure(fig, out)
    return out


if __name__ == "__main__":
    raise SystemExit(collect_and_write())
