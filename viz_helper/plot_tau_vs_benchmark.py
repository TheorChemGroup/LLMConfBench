#!/usr/bin/env python3

"""Square τ vs benchmark scores (release-scatter style, molecule 95% CI).

Reads scores from cohort_benchmark_scores.json (no live API calls).
Refresh that file with:

  python viz_helper/refresh_cohort_benchmark_scores.py

Writes GPQA Diamond and SciCode by default:

  python viz_helper/plot_tau_vs_benchmark.py
  python viz_helper/plot_tau_vs_benchmark.py --bench gpqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scipy.stats import linregress, spearmanr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scores import COHORT_BENCHMARK_SCORES, load_cohort_benchmark_scores
from plot_metrics_vs_release import collect_model_rows, scatter_model_point
from plot_tau_vs_release_square import (
    AXIS_LABEL_FONTSIZE,
    FLOOR_YLIM,
    _y_of,
    finish_square_axes,
    floor_tag,
    open_square_figure,
    place_square_legend,
    save_square_figure,
    ylim_with_references,
)

from config.models_config import (
    TAU_FLOOR,
    geom_conformers_dir,
    geom_seed_dirs,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models
from viz.plot_style import (
    ERROR_COLOR,
    REF_GFNFF,
    REF_MMFF,
    REF_MMFF_LINESTYLE,
    REF_RANDOM,
    REF_UFF,
    TEXT_COLOR,
)

FLOOR = TAU_FLOOR
Y_KEY = f"tau_f{FLOOR}"
SCORE_JITTER_RADIUS = 0.004
SCORE_JITTER_STEP = 0.003

AA_BENCHES: dict[str, tuple[str, str]] = {
    "gpqa": ("GPQA Diamond", "tau_vs_gpqa_square_w95"),
    "scicode": ("SciCode", "tau_vs_scicode_square_w95"),
}
GPQA_TREND_EXCLUDE: frozenset[str] = frozenset({
    "Qwen 3.5 27B",
    "Qwen 3.6 27B",
    "Qwen 3.5 35B A3B",
    "Qwen 3.6 35B A3B",
})


def _jitter_xs(xs: list[float]) -> list[float]:
    from plot_tau_vs_release_square import _jitter_xs as _j

    return _j(xs, radius=SCORE_JITTER_RADIUS, step=SCORE_JITTER_STEP)


def _ols_r2(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Return (slope, intercept, R², p-value) of OLS y ~ x."""
    fit = linregress(xs, ys)
    return (
        float(fit.slope),
        float(fit.intercept),
        float(fit.rvalue) ** 2,
        float(fit.pvalue),
    )


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _draw_trend(
    ax,
    xs: list[float],
    ys: list[float],
    *,
    linestyle: str = "-",
    color: str = TEXT_COLOR,
) -> tuple[float, float]:
    slope, intercept, r2, p = _ols_r2(xs, ys)
    x0, x1 = ax.get_xlim()
    ax.plot(
        [x0, x1],
        [intercept + slope * x0, intercept + slope * x1],
        color=color,
        linestyle=linestyle,
        linewidth=1.1,
        zorder=2,
        label="_nolegend_",
    )
    return r2, p


def write_tau_vs_aa(
    out: Path,
    models: list[dict],
    rand: dict,
    uff: dict | None,
    gfnff: dict | None,
    mmff: dict | None,
    scores_by_name: dict[str, float],
    *,
    xlabel: str,
    y_key: str = Y_KEY,
    ylabel: str = f"Conformers Ranking Accuracy (|ΔE| ≥ {FLOOR:.1f} kcal/mol)",
    legend_names: set[str] | None = None,
    second_trend_exclude: frozenset[str] | None = None,
) -> Path:
    plotted: list[tuple[str, float, float, float, float]] = []
    for m in models:
        name = m["name"]
        if y_key not in m or name not in scores_by_name:
            continue
        yerr = float(m.get(f"{y_key}_ci") or 0.0)
        alpha = 0.5 if m.get("pilot_partial") else 1.0
        plotted.append((name, scores_by_name[name], float(m[y_key]), yerr, alpha))
    if len(plotted) < 3:
        raise SystemExit(f"Need ≥3 models with {xlabel} and {y_key}; got {len(plotted)}")

    xs = _jitter_xs([p[1] for p in plotted])
    score_x = [p[1] for p in plotted]
    tau_y = [p[2] for p in plotted]

    fig, ax, legend_ax = open_square_figure()

    y_rand = 0.0
    ax.axhline(
        y_rand,
        color=REF_RANDOM, linestyle="--", linewidth=1.1, zorder=0,
    )
    y_uff = _y_of(uff, y_key)
    if y_uff is not None:
        ax.axhline(y_uff, color=REF_UFF, linestyle=":", linewidth=1.3, zorder=1)
    y_gfnff = _y_of(gfnff, y_key)
    if y_gfnff is not None:
        ax.axhline(y_gfnff, color=REF_GFNFF, linestyle="-.", linewidth=1.3, zorder=1)
    y_mmff = _y_of(mmff, y_key)
    if y_mmff is not None:
        ax.axhline(
            y_mmff, color=REF_MMFF, linestyle=REF_MMFF_LINESTYLE, linewidth=1.3, zorder=1,
        )

    for (name, _, y, yerr, alpha), x in zip(plotted, xs, strict=True):
        if yerr > 0:
            ax.errorbar(
                x, y, yerr=yerr, fmt="none",
                ecolor=ERROR_COLOR, elinewidth=0.5, capsize=0,
                alpha=0.15 * alpha, zorder=4, clip_on=False,
            )
        scatter_model_point(ax, name, x, y, uniform_size=True, alpha=alpha)

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    finish_square_axes(
        ax, ylabel,
        ylim_with_references(FLOOR_YLIM, y_rand, y_uff, y_gfnff, y_mmff),
    )
    xmin, xmax = min(xs), max(xs)
    pad = max(0.02, 0.06 * (xmax - xmin))
    ax.set_xlim(xmin - pad, xmax + pad)
    r2, pval = _draw_trend(ax, score_x, tau_y, linestyle="-")
    rho, rho_p = spearmanr(score_x, tau_y)
    print(
        f"{xlabel}  n={len(plotted)}  R²={r2:.3f}  p={_fmt_p(pval)}  "
        f"Spearman ρ={float(rho):.3f}  p={_fmt_p(float(rho_p))}"
    )
    if second_trend_exclude:
        sub_x = [row[1] for row in plotted if row[0] not in second_trend_exclude]
        sub_y = [row[2] for row in plotted if row[0] not in second_trend_exclude]
        dropped = [row[0] for row in plotted if row[0] in second_trend_exclude]
        if len(sub_x) >= 2:
            r2b, pb = _draw_trend(ax, sub_x, sub_y, linestyle="--", color="#A21717")
            print(
                f"{xlabel}  n={len(sub_x)}  R²={r2b:.3f}  p={_fmt_p(pb)}  "
                f"(excl. {', '.join(dropped) or 'none plotted'})"
            )

    names = {p[0] for p in plotted} if legend_names is None else legend_names
    place_square_legend(
        legend_ax,
        names,
        has_uff=False,
        has_gfnff=False,
        marker_by="reasoning",
        size_by_params=False,
    )
    save_square_figure(fig, out)
    return out


def _write_one_bench(
    bench: str,
    *,
    models: list[dict],
    rand: dict,
    uff: dict | None,
    gfnff: dict | None,
    mmff: dict | None,
    scores_by_display: dict[str, dict[str, float]],
    out: Path | None,
) -> Path:
    if bench not in AA_BENCHES:
        raise SystemExit(f"Unknown bench {bench!r}; choose from {list(AA_BENCHES)}")
    xlabel, stem = AA_BENCHES[bench]
    scores_by_name: dict[str, float] = {}
    missing: list[str] = []
    for m in models:
        name = m["name"]
        score = (scores_by_display.get(name) or {}).get(bench)
        if score is None:
            missing.append(name)
            continue
        scores_by_name[name] = float(score)
    if missing:
        print(f"no {xlabel}:", ", ".join(missing))

    if out is None:
        tag = floor_tag(FLOOR)
        out = Path("figures/release_scatter/r2scan3c") / f"{stem}_{tag}.png"
    exclude = GPQA_TREND_EXCLUDE if bench == "gpqa" else None
    return write_tau_vs_aa(
        out, models, rand, uff, gfnff, mmff, scores_by_name,
        xlabel=xlabel, second_trend_exclude=exclude,
    )


def collect_and_write(
    out: Path | None = None,
    *,
    forcefields: bool = True,
    benches: list[str] | None = None,
    scores_path: Path = COHORT_BENCHMARK_SCORES,
) -> list[Path]:
    chosen = list(benches) if benches else list(AA_BENCHES)
    unknown = [b for b in chosen if b not in AA_BENCHES]
    if unknown:
        raise SystemExit(f"Unknown bench {unknown}; choose from {list(AA_BENCHES)}")
    if out is not None and len(chosen) != 1:
        raise SystemExit("--out is only valid with a single --bench")

    bundle_dir = geom_seed_dirs()[0][1]
    deltae_override = load_deltae_override(resolve_deltae_json("r2scan3c"))
    models, rand, uff, gfnff, mmff = collect_model_rows(
        bundle_dir,
        geom_conformers_dir(),
        deltae_override=deltae_override,
        three_seeds=True,
        forcefields="all" if forcefields else "none",
        floors=[FLOOR],
        model_files=dict(complete_geom_models(
            deltae_override=deltae_override,
            warn_unregistered=False,
        )),
    )
    scores_by_display = load_cohort_benchmark_scores(scores_path)
    written: list[Path] = []
    for bench in chosen:
        written.append(_write_one_bench(
            bench,
            models=models, rand=rand, uff=uff, gfnff=gfnff, mmff=mmff,
            scores_by_display=scores_by_display,
            out=out if len(chosen) == 1 else None,
        ))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench",
        nargs="*",
        choices=list(AA_BENCHES),
        default=None,
        help="Benchmark key(s) (default: gpqa scicode)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG for a single --bench "
             "(default: figures/release_scatter/r2scan3c/tau_vs_<bench>_square_w95_2.png)",
    )
    ap.add_argument(
        "--no-forcefields",
        action="store_false",
        dest="forcefields",
        help="Omit UFF / GFN-FF reference lines",
    )
    ap.add_argument(
        "--scores",
        type=Path,
        default=COHORT_BENCHMARK_SCORES,
        help=f"Cohort benchmark scores JSON (default: {COHORT_BENCHMARK_SCORES.name})",
    )
    ap.set_defaults(forcefields=True)
    args = ap.parse_args()
    collect_and_write(
        args.out,
        forcefields=args.forcefields,
        benches=args.bench,
        scores_path=args.scores,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
