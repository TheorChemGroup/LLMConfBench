#!/usr/bin/env python3

"""Facet figure: τ @ 2.0 vs every cohort benchmark score (one panel per predictor).

    python viz_helper/benchmark_corr_table.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scores import COHORT_BENCHMARK_SCORES, parse_row_scores
from plot_metrics_vs_release import (
    collect_model_rows,
    scatter_model_point,
    total_api_cost_by_display,
)
from plot_tau_vs_arcagi import DEFAULT_ARCAGI
from plot_tau_vs_benchmark import (
    FLOOR,
    GPQA_TREND_EXCLUDE,
    Y_KEY,
    _draw_trend,
    _jitter_xs,
)
from plot_tau_vs_release_square import (
    AXIS_LABEL_FONTSIZE,
    FLOOR_YLIM,
    TICK_FONTSIZE,
    _date_to_x,
    _legend_handles,
    _y_of,
    apply_floor_ylim,
    floor_tag,
    ylim_with_references,
)

from config.models_config import (
    geom_conformers_dir,
    geom_seed_dirs,
    models_by_id,
    params_total_for_display,
    release_dates_by_display,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models
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

DEFAULT_FIGURE = Path("figures/release_scatter/r2scan3c") / (
    f"tau_vs_benchmarks_facet_{floor_tag(FLOOR)}.png"
)
DEFAULT_TSV = Path("figures/release_scatter/r2scan3c") / (
    f"tau_benchmark_correlations_{floor_tag(FLOOR)}.tsv"
)

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


def _tau_by_name(models: list[dict], y_key: str = Y_KEY) -> dict[str, float]:
    return {m["name"]: float(m[y_key]) for m in models if y_key in m}


def _load_cohort_scores(
    scores_path: Path = COHORT_BENCHMARK_SCORES,
) -> dict[str, dict[str, float]]:
    short_to_display = {m.short: m.display for m in models_by_id().values()}
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    by_display: dict[str, dict[str, float]] = {}
    for row in payload.get("benchmark_results", []):
        short = row.get("model")
        if not short:
            continue
        display = short_to_display.get(short)
        if display is None:
            continue
        parsed = parse_row_scores(row)
        if parsed:
            by_display[display] = parsed
    return by_display


def _load_arc_scores(arcagi_path: Path = DEFAULT_ARCAGI) -> dict[str, dict[str, float]]:
    from benchmark_scores import parse_pct_score

    short_to_display = {m.short: m.display for m in models_by_id().values()}
    payload = json.loads(arcagi_path.read_text(encoding="utf-8"))
    by_display: dict[str, dict[str, float]] = {}
    for row in payload.get("benchmark_results", []):
        short = row.get("model")
        if not short:
            continue
        display = short_to_display.get(short)
        if display is None:
            continue
        parsed = {
            k: parse_pct_score(row.get(k))
            for k in ("arc_agi_1", "arc_agi_2", "arc_agi_3")
        }
        by_display[display] = {k: v for k, v in parsed.items() if v is not None}
    return by_display


def collect_scatter_data(
    *,
    scores_path: Path = COHORT_BENCHMARK_SCORES,
    arcagi_path: Path = DEFAULT_ARCAGI,
    baseline: str = "r2scan3c",
) -> tuple[list[tuple[str, list[float], list[float], list[str]]], tuple]:
    """Per-benchmark scatter data: (label, xs, ys, names) for the facet figure."""
    bundle_dir = geom_seed_dirs()[0][1]
    deltae_override = load_deltae_override(resolve_deltae_json(baseline))
    models, rand, uff, gfnff, mmff = collect_model_rows(
        bundle_dir,
        geom_conformers_dir(),
        deltae_override=deltae_override,
        three_seeds=True,
        forcefields="all",
        floors=[FLOOR],
        model_files=dict(
            complete_geom_models(deltae_override=deltae_override, warn_unregistered=False)
        ),
    )
    tau = _tau_by_name(models)
    cohort_scores = _load_cohort_scores(scores_path)
    arc_scores = _load_arc_scores(arcagi_path)
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
    panels, (_rand, uff, gfnff, mmff) = collect_scatter_data(
        scores_path=scores_path, arcagi_path=arcagi_path, baseline=baseline,
    )
    ylabel = f"Conformers Ranking Accuracy (|ΔE| ≥ {FLOOR:.1f} kcal/mol)"
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    refs = (
        (0.0, REF_RANDOM, "--"),
        (_y_of(uff, Y_KEY), REF_UFF, ":"),
        (_y_of(gfnff, Y_KEY), REF_GFNFF, "-."),
        (_y_of(mmff, Y_KEY), REF_MMFF, REF_MMFF_LINESTYLE),
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

    fig.supylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    present = {name for _label, _xs, _ys, names in panels for name in names}
    handles = _legend_handles(
        present, has_uff=False, has_gfnff=False, has_mmff=False,
        marker_by="reasoning", size_by_params=False,
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=FONTSIZE, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save_fig(fig, out or DEFAULT_FIGURE)


def write_corr_tsv(
    out: Path | None = None,
    *,
    scores_path: Path = COHORT_BENCHMARK_SCORES,
    arcagi_path: Path = DEFAULT_ARCAGI,
    baseline: str = "r2scan3c",
) -> Path:
    """R² / p / n of τ vs every predictor, one row per benchmark."""
    panels, _ = collect_scatter_data(
        scores_path=scores_path, arcagi_path=arcagi_path, baseline=baseline,
    )
    lines = ["benchmark\tn\tr2\tp_value"]
    print(f"{'Benchmark':<28s} {'n':>3s}  {'R²':>6s}  {'p-value':>10s}")
    for label, xs, ys, names in panels:
        fit = linregress(xs, ys)
        r2 = float(fit.rvalue) ** 2
        p = float(fit.pvalue)
        p_s = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        r2_s = f"{r2:.3f}"
        if label.startswith("GPQA"):
            keep = [(x, y) for x, y, n in zip(xs, ys, names)
                    if n not in GPQA_TREND_EXCLUDE]
            if len(keep) >= 3:
                decl = linregress([k[0] for k in keep], [k[1] for k in keep])
                r2_s = f"{r2:.3f} ({float(decl.rvalue) ** 2:.3f} declustered)"
        lines.append(f"{label}\t{len(xs)}\t{r2_s}\t{p_s}")
        print(f"{label:<28s} {len(xs):3d}  {r2_s:>28s}  {p_s:>10s}")
    out = out or DEFAULT_TSV
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_FIGURE)
    ap.add_argument("--out-tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--scores", type=Path, default=COHORT_BENCHMARK_SCORES)
    ap.add_argument("--arcagi", type=Path, default=DEFAULT_ARCAGI)
    ap.add_argument("--baseline", default="r2scan3c")
    args = ap.parse_args()
    write_corr_tsv(
        args.out_tsv,
        scores_path=args.scores,
        arcagi_path=args.arcagi,
        baseline=args.baseline,
    )
    write_facet_figure(
        args.out,
        scores_path=args.scores,
        arcagi_path=args.arcagi,
        baseline=args.baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
