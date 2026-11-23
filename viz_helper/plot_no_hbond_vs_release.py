#!/usr/bin/env python3

"""CRA on the 2739 0-vs-0 H-bond pairs vs release year (main-figure style).

No-H-bond pairs: neither conformer forms an intramolecular H-bond
(O/N acceptor, <= 2.5 A, angle >= 120 deg), and the pair differs by at least
2.0 kcal/mol against r2SCAN-3c.
Year on x, mean per-molecule CRA on y, force fields as horizontal reference lines.

    python viz_helper/plot_no_hbond_vs_release.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoy_common import (
    BUNDLES,
    FLOOR,
    bench_molecules,
    hb_count_map,
    hbond_pair_tables,
    load_r2scan,
    override_r2scan3c,
    score_pairs_ff,
    score_pairs_letters,
)
from hb_rule_baseline import draw as draw_hb_rule
from metrics.grouped_metrics import cra_macro  # noqa: E402
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
    place_square_legend,
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
OUT = Path("figures/cra_no_hbond_vs_release.png")
YLIM = (-0.1, 1.0)
YTICKS = np.arange(-0.1, 1.01, 0.1)


def collect_and_write(out: Path | None = None) -> Path:
    out = Path(out) if out is not None else OUT
    r2scan = load_r2scan()
    bench = bench_molecules()
    counts = hb_count_map(r2scan, bench)
    table = hbond_pair_tables(r2scan, counts)["noHB"]
    n_pairs = sum(len(v) for v in table.values())

    override = override_r2scan3c()
    bdirs = [GEOM / b for b in BUNDLES]
    dates = release_dates_by_display()

    pts, ff_pts = [], []
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            continue
        tau, half, _ = cra_macro(score_pairs_letters(bdirs, pred, {"noHB": table}, override)["noHB"])
        pts.append((display, dates[display], tau, half))
    for name, getter in (("UFF", get_uff_ranking), ("GFN-FF", get_gfnff_ranking),
                         ("MMFF94", get_mmff94_ranking)):
        tau, _, _ = cra_macro(score_pairs_ff(getter, {"noHB": table}, r2scan)["noHB"])
        ff_pts.append((name, tau))
    print(f"no-H-bond pairs: {n_pairs} (|dE| >= {FLOOR:g} kcal/mol), models: {len(pts)}, "
          f"FF: {[(n, f'{t:+.3f}') for n, t in ff_pts]}", flush=True)

    present = [p[0] for p in pts]
    xs = _jitter_xs([_date_to_x(p[1]) for p in pts])
    fig, ax, legend_ax = open_square_figure()
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
    draw_hb_rule(ax, table, counts)
    ax.axhline(0.0, color="grey", ls=":", lw=0.8, zorder=1)
    _set_date_axis(ax)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    style_ax(ax, square=False)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("CRA across pairs without H-bonds", fontsize=10)
    place_square_legend(legend_ax, set(present), marker_by="reasoning")
    save_square_figure(fig, out)
    return out


if __name__ == "__main__":
    collect_and_write()
