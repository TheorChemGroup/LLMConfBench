#!/usr/bin/env python3

"""Facet figure: τ @ 2.0 vs every cohort benchmark score (one panel per predictor).

    python viz_helper/benchmark_corr_table.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scores import COHORT_BENCHMARK_SCORES, load_cohort_benchmark_scores
from plot_eqhb_vs_benchmarks import equal_hb_rows
from plot_metrics_vs_release import (
    scatter_model_point,
    total_api_cost_by_display,
)
from plot_tau_vs_arcagi import DEFAULT_ARCAGI, load_arc_by_display
from plot_tau_vs_benchmark import _draw_trend, _jitter_xs
from plot_tau_vs_release_square import (
    AXIS_LABEL_FONTSIZE,
    FLOOR_YLIM,
    TICK_FONTSIZE,
    _date_to_x,
    _legend_handles,
    _y_of,
    apply_floor_ylim,
    ylim_with_references,
)

from config.models_config import (
    params_total_for_display,
    release_dates_by_display,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.plot_style import (
    FONTSIZE,
    REF_GFNFF,
    REF_MMFF,
    REF_MMFF_LINESTYLE,
    REF_RANDOM,
    REF_UFF,
    plt,
    save_fig,
    style_ax,
)

DEFAULT_FIGURE = Path("figures") / "cra_vs_benchmarks_facet.png"


def _equal_hb_context(
    baseline: str = "r2scan3c",
) -> tuple[dict[str, float], tuple, int]:
    """Equal-H-bond CRA per model + FF reference dicts + pair count."""
    rows, ff, n_pairs, _ = equal_hb_rows(baseline)
    tau = {r["name"]: float(r["eqhb"]) for r in rows}
    refs = ({}, ff["UFF"], ff["GFN-FF"], ff["MMFF94"])
    return tau, refs, n_pairs

COHORT_LABELS: dict[str, str] = {
    "agentic": "Agentic",
    "aime": "AIME",
    "coding": "Coding",
    "gpqa": "GPQA Diamond",
    "hle": "HLE",
    "ifbench": "IFBench",
    "intelligence": "Intelligence",
    "lcr": "LCR",
    "livecodebench": "LiveCodeBench",
    "math": "Math",
    "mmlu_pro": "MMLU-Pro",
    "scicode": "SciCode",
    "terminalbench": "TerminalBench",
}

ARC_LABELS: dict[str, str] = {
    "arc_agi_1": "ARC-AGI-1",
    "arc_agi_2": "ARC-AGI-2",
    "arc_agi_3": "ARC-AGI-3",
}


def collect_scatter_data(
    *,
    scores_path: Path = COHORT_BENCHMARK_SCORES,
    arcagi_path: Path = DEFAULT_ARCAGI,
    baseline: str = "r2scan3c",
    tau_by_name: dict[str, float],
    refs: tuple,
) -> tuple[list[tuple[str, list[float], list[float], list[str]]], tuple]:
    """Per-benchmark scatter data: (label, xs, ys, names) for the facet figure."""
    deltae_override = load_deltae_override(resolve_deltae_json(baseline))
    tau = tau_by_name
    rand, uff, gfnff, mmff = refs
    cohort_scores = load_cohort_benchmark_scores(scores_path)
    arc_scores = load_arc_by_display(arcagi_path, verbose=False)
    dates = release_dates_by_display()
    cost_by_name, _, _ = total_api_cost_by_display(deltae_override=deltae_override)

    panels: list[tuple[str, list[float], list[float], list[str]]] = []

    def add(label: str, x_of) -> None:
        xs, ys, names = [], [], []
        for name, tau_val in tau.items():
            x = x_of(name)
            if x is None:
                continue
            xs.append(float(x))
            ys.append(float(tau_val))
            names.append(name)
        if len(xs) >= 3:
            panels.append((label, xs, ys, names))

    bench_keys = sorted(
        {k for scores in cohort_scores.values() for k in scores},
        key=lambda k: COHORT_LABELS.get(k, k),
    )
    for key in bench_keys:
        add(COHORT_LABELS.get(key, key),
            lambda name, key=key: (cohort_scores.get(name) or {}).get(key))
    for json_key, label in ARC_LABELS.items():
        add(label, lambda name, json_key=json_key: (arc_scores.get(name) or {}).get(json_key))
    add("Release date", lambda name: _date_to_x(dates[name]) if name in dates else None)

    def _log_params(name: str) -> float | None:
        params = params_total_for_display(name)
        return math.log10(params) if params and params > 0 else None

    add("Total parameters (log10)", _log_params)
    add("Total API cost ($)", lambda name: cost_by_name.get(name))
    return panels, (rand, uff, gfnff, mmff)


def write_facet_figure(
    out: Path | None = None,
    *,
    scores_path: Path = COHORT_BENCHMARK_SCORES,
    arcagi_path: Path = DEFAULT_ARCAGI,
    baseline: str = "r2scan3c",
) -> Path:
    """One facet per benchmark: τ (y) vs benchmark score (x), with OLS trend."""
    tau, refs, _n_pairs = _equal_hb_context(baseline)
    panels, (_rand, uff, gfnff, mmff) = collect_scatter_data(
        scores_path=scores_path, arcagi_path=arcagi_path, baseline=baseline,
        tau_by_name=tau, refs=refs,
    )
    ylabel = "CRA across fair pairs"
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    refs = (
        (0.0, REF_RANDOM, "--"),
        (_y_of(uff, "eqhb"), REF_UFF, ":"),
        (_y_of(gfnff, "eqhb"), REF_GFNFF, "-."),
        (_y_of(mmff, "eqhb"), REF_MMFF, REF_MMFF_LINESTYLE),
    )
    ylim = ylim_with_references(FLOOR_YLIM, *(value for value, _c, _ls in refs))
    for ax, (label, xs, ys, names) in zip(axes, panels):
        for value, color, ls in refs:
            if value is not None:
                ax.axhline(value, color=color, linestyle=ls, linewidth=1.0, zorder=0)
        for name, xj, y in zip(names, _jitter_xs(xs), ys):
            scatter_model_point(ax, name, xj, y, uniform_size=True)
        _draw_trend(ax, xs, ys)
        ax.set_title(label, fontsize=FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        style_ax(ax, square=True)
        apply_floor_ylim(ax, ylim)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.supylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, x=0.030)
    present = {name for _label, _xs, _ys, names in panels for name in names}
    handles = _legend_handles(
        present, marker_by="reasoning", size_by_params=False,
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=FONTSIZE, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save_fig(fig, out or DEFAULT_FIGURE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_FIGURE)
    ap.add_argument("--scores", type=Path, default=COHORT_BENCHMARK_SCORES)
    ap.add_argument("--arcagi", type=Path, default=DEFAULT_ARCAGI)
    ap.add_argument("--baseline", default="r2scan3c")
    args = ap.parse_args()
    write_facet_figure(
        args.out,
        scores_path=args.scores,
        arcagi_path=args.arcagi,
        baseline=args.baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
