#!/usr/bin/env python3

"""Square Kendall-τ scatter (no per-model labels).

Default: τ vs release year. Thinking/reasoning models: circles. Others:
triangles. Fill colour is the per-model colour from models.yaml. All markers
are the same size. Date axis is linear in calendar years (ticks 2025–2027)
on a 3:2 canvas. Models that share a release window are jittered along x.

`--x params`: τ vs log10(total parameters). Skip models with unpublished
params. Marker encodes architecture (dense/MoE/unpublished). Colour is family.

`--size-by-params`: keep the date axis; scale marker area with log10(total
params). Closed-weight models (`class: closed`) are omitted. Unpublished
params keep the default marker size.

`--only os|closed`: plot only open-weight or closed-weight models.

  python viz_helper/plot_tau_vs_release_square.py
  python viz_helper/plot_tau_vs_release_square.py --x params
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import (
    LEGEND_MARKERSIZE,
    MARKER_DEFAULT,
    MARKER_REASONING,
    MARKER_SIZE,
    VENDOR_FAMILIES,
    arch_for_display,
    baseline_ids,
    family_for_display,
    family_legend_color,
    geom_conformers_dir,
    geom_seed_dirs,
    is_closed_display,
    is_os_display,
    marker_for_arch,
    models_by_id,
    params_total_for_display,
    release_dates_by_display,
    resolve_deltae_json,
    scatter_size_for_params,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models
from viz.plot_style import (
    ERROR_COLOR,
    FONTSIZE,
    FONTSIZE_LABEL,
    ONE_HALF_COLUMN_WIDTH,
    REF_GFNFF,
    REF_MMFF,
    REF_MMFF_LINESTYLE,
    REF_RANDOM,
    REF_UFF,
    TEXT_COLOR,
    plt,
    style_ax,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_metrics_vs_release import collect_model_rows, scatter_model_point

FIGSIZE = (ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH)
SAVE_DPI = 2000
FIGSIZE_DATE = (ONE_HALF_COLUMN_WIDTH * 4.0 / 2.0, ONE_HALF_COLUMN_WIDTH)
AXIS_LABEL_FONTSIZE = FONTSIZE_LABEL
TICK_FONTSIZE = FONTSIZE
LEGEND_FONTSIZE = FONTSIZE
FLOOR_YLIM = (-0.1, 0.8)

SQUARE_AXES = (0.24, 0.355, 0.62, 0.62)
SQUARE_LEGEND = (0.04, 0.008, 0.94, 0.228)
DATE_AXES = (0.24, 0.29, 0.62, 0.69)
DATE_LEGEND = (0.0, 0.0, 1.0, 0.20)
SQUARE_YLABEL_X = -0.20
DATE_YLABEL_X = -0.10
SQUARE_XLABEL_Y = -0.12

_X_YEAR0 = datetime(2024, 1, 1)  # noqa: DTZ001
_X_LEFT = datetime(2024, 7, 1)  # noqa: DTZ001
_X_RIGHT = datetime(2027, 4, 1)  # noqa: DTZ001
_X_TICKS: tuple[tuple[datetime, str], ...] = (
    (datetime(2025, 1, 1), "2025"),  # noqa: DTZ001
    (datetime(2026, 1, 1), "2026"),  # noqa: DTZ001
    (datetime(2027, 1, 1), "2027"),  # noqa: DTZ001
)


def _date_to_x(dt: datetime) -> float:
    """Map calendar time to axis x in years since 2024-01-01."""
    return (dt - _X_YEAR0).days / 365.25


_JITTER_RADIUS = 0.10
_JITTER_STEP = 0.035
_PARAMS_JITTER_RADIUS = 0.04
_PARAMS_JITTER_STEP = 0.03
_PARAM_TICKS_B = (30.0, 100.0, 300.0, 1000.0, 3000.0)
_SIZE_LEGEND_B = (30.0, 300.0, 3000.0)
_ARCH_LEGEND: tuple[tuple[str | None, str], ...] = (
    ("dense", "Dense"),
    ("moe", "MoE"),
    ("distill", "Distill"),
    (None, "Unpublished"),
)


def _jitter_xs(
    xs: list[float],
    *,
    radius: float = _JITTER_RADIUS,
    step: float = _JITTER_STEP,
) -> list[float]:
    """Spread points that fall in the same window, keeping original order."""
    n = len(xs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if abs(xs[i] - xs[j]) <= radius:
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = list(xs)
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: xs[i])
        mean_x = sum(xs[i] for i in members) / len(members)
        k = len(members)
        for t, i in enumerate(members):
            out[i] = mean_x + (t - (k - 1) / 2.0) * step
    return out


def floor_tag(floor: float) -> str:
    """0.5 → 05, 1.0 → 1, 1.5 → 15."""
    if abs(floor - 0.5) < 1e-9:
        return "05"
    if abs(floor - 1.0) < 1e-9:
        return "1"
    if abs(floor - 1.5) < 1e-9:
        return "15"
    return f"{floor:g}".replace(".", "")


def _y_of(row: dict | None, key: str) -> float | None:
    if row is None or key not in row:
        return None
    val = row[key]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _style_handle(
    *,
    marker: str,
    facecolor: str,
    label: str,
    markersize: float = LEGEND_MARKERSIZE,
) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker=marker,
        color="w",
        markerfacecolor=facecolor,
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=markersize,
        linestyle="None",
        label=label,
    )


def _legend_markersize(scatter_s: float) -> float:
    return math.sqrt(scatter_s)


def _legend_handles(
    present_names: set[str],
    *,
    has_uff: bool,
    has_gfnff: bool,
    has_mmff: bool = False,
    marker_by: str = "reasoning",
    size_by_params: bool = False,
) -> list[Line2D]:
    handles: list[Line2D] = [
        Line2D(
            [0], [0],
            color=REF_RANDOM,
            linestyle="--",
            linewidth=1.1,
            label="Random",
        ),
    ]
    if has_uff:
        handles.append(
            Line2D(
                [0], [0],
                color=REF_UFF,
                linestyle=":",
                linewidth=1.3,
                label="UFF",
            )
        )
    if has_gfnff:
        handles.append(
            Line2D(
                [0], [0],
                color=REF_GFNFF,
                linestyle="-.",
                linewidth=1.3,
                label="GFN-FF",
            )
        )
    if has_mmff:
        handles.append(
            Line2D(
                [0], [0],
                color=REF_MMFF,
                linestyle=REF_MMFF_LINESTYLE,
                linewidth=1.3,
                label="MMFF94",
            )
        )
    if marker_by == "arch":
        for arch, label in _ARCH_LEGEND:
            if arch not in ("dense", "moe"):
                continue
            handles.append(
                _style_handle(
                    marker=marker_for_arch(arch),
                    facecolor="#6b7280",
                    label=label,
                )
            )
    else:
        handles.append(
            _style_handle(
                marker=MARKER_REASONING,
                facecolor="#6b7280",
                label="Reasoning",
            )
        )
        handles.append(
            _style_handle(
                marker=MARKER_DEFAULT,
                facecolor="#6b7280",
                label="Non-reasoning",
            )
        )
    if size_by_params:
        for params_b in _SIZE_LEGEND_B:
            handles.append(
                _style_handle(
                    marker=MARKER_REASONING,
                    facecolor="#6b7280",
                    label=f"{params_b:g}B",
                    markersize=_legend_markersize(
                        scatter_size_for_params(params_b)
                    ),
                )
            )
        if any(params_total_for_display(n) is None for n in present_names):
            handles.append(
                _style_handle(
                    marker=MARKER_REASONING,
                    facecolor="#6b7280",
                    label="params unpublished",
                    markersize=_legend_markersize(MARKER_SIZE),
                )
            )
    for fam in VENDOR_FAMILIES:
        if not any(family_for_display(n) == fam for n in present_names):
            continue
        handles.append(
            _style_handle(
                marker=MARKER_REASONING,
                facecolor=family_legend_color(fam, present_names),
                label=fam,
            )
        )
    return handles


def apply_floor_ylim(ax: plt.Axes, ylim: tuple[float, float] = FLOOR_YLIM) -> None:
    lo, hi = ylim
    ax.set_ylim(lo, hi)
    ticks = [round(lo + i * 0.1, 10) for i in range(round((hi - lo) / 0.1) + 1)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def ylim_with_references(
    base_ylim: tuple[float, float],
    *reference_values: float | None,
) -> tuple[float, float]:
    """Raise the upper y-limit so no reference line is clipped off the axes."""
    lo, hi = base_ylim
    finite = [v for v in reference_values if v is not None and math.isfinite(v)]
    if finite:
        hi = max(hi, round(max(finite) + 0.05, 1))
    return lo, hi


def open_square_figure(
    *,
    figsize: tuple[float, float] | None = None,
    axes_rect: tuple[float, float, float, float] = SQUARE_AXES,
    legend_rect: tuple[float, float, float, float] = SQUARE_LEGEND,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Identical axes rect; figsize is square unless the date scatter passes 3:2."""
    fig = plt.figure(figsize=figsize or FIGSIZE)
    ax = fig.add_axes(axes_rect)
    legend_ax = fig.add_axes(legend_rect)
    legend_ax.axis("off")
    return fig, ax, legend_ax


def finish_square_axes(
    ax: plt.Axes,
    ylabel: str,
    ylim: tuple[float, float] = FLOOR_YLIM,
    *,
    ylabel_x: float | None = None,
) -> None:
    """Shared y-limits, ticks, and label anchors so reference lines line up."""
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    style_ax(ax, square=False)
    apply_floor_ylim(ax, ylim)
    ax.yaxis.set_label_coords(SQUARE_YLABEL_X if ylabel_x is None else ylabel_x, 0.5)
    ax.xaxis.set_label_coords(0.5, SQUARE_XLABEL_Y)


def place_square_legend(
    legend_ax: plt.Axes,
    present_names: set[str],
    *,
    has_uff: bool,
    has_gfnff: bool,
    has_mmff: bool = False,
    marker_by: str = "reasoning",
    size_by_params: bool = False,
    anchor_bottom: bool = False,
) -> None:
    legend_ax.legend(
        handles=_legend_handles(
            present_names,
            has_uff=has_uff,
            has_gfnff=has_gfnff,
            has_mmff=has_mmff,
            marker_by=marker_by,
            size_by_params=size_by_params,
        ),
        loc="lower center" if anchor_bottom else "upper center",
        bbox_to_anchor=(0.5, 0.0) if anchor_bottom else (0.5, 1.0),
        ncol=4,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
        borderpad=0.0,
    )


def save_square_figure(fig: plt.Figure, out: Path) -> None:
    """Fixed canvas (no tight crop) so panels share the same pixel layout."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def display_for_output_tag(tag: str) -> str:
    """Resolve a model id or answers_<tag>.jsonl stem to its display name."""
    specs = models_by_id()
    if tag in specs and specs[tag].answers is not None:
        return specs[tag].display
    filename = f"answers_{tag}.jsonl"
    matches = [spec.display for spec in specs.values() if spec.answers == filename]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"Unknown --output-tag {tag!r}; expected a model id or configured answers-file tag."
    )


def _x_value(
    name: str,
    dates: dict[str, datetime],
    x_mode: str,
) -> float | None:
    if x_mode == "params":
        params = params_total_for_display(name)
        if params is None or params <= 0:
            return None
        return math.log10(params)
    if name not in dates:
        return None
    return _date_to_x(dates[name])


def _set_date_axis(ax: plt.Axes) -> None:
    ax.set_xticks([_date_to_x(dt) for dt, _ in _X_TICKS])
    ax.set_xticklabels([lab for _, lab in _X_TICKS])
    ax.minorticks_off()
    ax.set_xlim(_date_to_x(_X_LEFT), _date_to_x(_X_RIGHT))
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)


def _set_params_axis(ax: plt.Axes, xs: list[float]) -> None:
    ticks = [math.log10(v) for v in _PARAM_TICKS_B]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in _PARAM_TICKS_B])
    if xs:
        pad = 0.12
        lo = min(min(xs) - pad, ticks[0] - 0.08)
        hi = max(max(xs) + pad, ticks[-1] + 0.08)
        ax.set_xlim(lo, hi)
    ax.set_xlabel("Total parameters (B)", fontsize=AXIS_LABEL_FONTSIZE)


_FLOOR_STEM_EXTRAS = ("_params", "_closed", "_os")


def with_floor_stem(path: Path, floor: float) -> Path:
    """Insert the floor tag before a trailing class/params suffix if present."""
    stem = path.stem
    extra = ""
    for suffix in _FLOOR_STEM_EXTRAS:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            extra = suffix
            break
    return path.with_name(f"{stem}_{floor_tag(floor)}{extra}{path.suffix}")


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
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    *,
    linestyle: str = "-",
) -> tuple[float, float, float]:
    """Draw OLS trend; return (slope, R², p-value)."""
    slope, intercept, r2, p = _ols_r2(xs, ys)
    x0, x1 = ax.get_xlim()
    ax.plot(
        [x0, x1],
        [intercept + slope * x0, intercept + slope * x1],
        color=TEXT_COLOR,
        linestyle=linestyle,
        linewidth=1.1,
        zorder=2,
        label="_nolegend_",
    )
    return slope, r2, p


def _include_model(
    name: str,
    *,
    size_by_params: bool,
    only: str | None,
) -> bool:
    if only == "os":
        return is_os_display(name)
    if only == "closed":
        return is_closed_display(name)
    if size_by_params:
        return not is_closed_display(name)
    return True


def plot_square(
    ax: plt.Axes,
    models: list[dict],
    rand: dict,
    uff: dict | None,
    gfnff: dict | None,
    mmff: dict | None,
    dates: dict[str, datetime],
    *,
    y_key: str = "tau",
    ylabel: str = "Conformers Ranking Accuracy (|ΔE| ≥ 2.0 kcal/mol)",
    ylim: tuple[float, float] | None = None,
    show_ci: bool = False,
    x_mode: str = "date",
    marker_by: str = "reasoning",
    size_by_params: bool = False,
    only: str | None = None,
    ylabel_x: float | None = None,
) -> set[str]:
    y_rand = 0.0
    ax.axhline(y_rand, color=REF_RANDOM, linestyle="--", linewidth=1.1, zorder=0)
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

    plotted: list[tuple[str, float, float, float, float]] = []
    for m in models:
        name = m["name"]
        if y_key not in m:
            continue
        if not _include_model(
            name, size_by_params=size_by_params, only=only,
        ):
            continue
        x = _x_value(name, dates, x_mode)
        if x is None:
            continue
        yerr = float(m.get(f"{y_key}_ci") or 0.0)
        alpha = 0.5 if m.get("pilot_partial") else 1.0
        plotted.append((name, x, float(m[y_key]), yerr, alpha))
    if x_mode == "params":
        xs = _jitter_xs(
            [p[1] for p in plotted],
            radius=_PARAMS_JITTER_RADIUS,
            step=_PARAMS_JITTER_STEP,
        )
    else:
        xs = _jitter_xs([p[1] for p in plotted])
    present: set[str] = set()
    for (name, _, y, yerr, alpha), x in zip(plotted, xs, strict=True):
        if show_ci and yerr > 0:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=ERROR_COLOR,
                elinewidth=0.5,
                capsize=0,
                alpha=0.15 * alpha,
                zorder=4,
                clip_on=False,
            )
        marker = None
        if marker_by == "arch":
            marker = marker_for_arch(arch_for_display(name))
        size = None
        if size_by_params:
            size = scatter_size_for_params(params_total_for_display(name))
        scatter_model_point(
            ax,
            name,
            x,
            y,
            uniform_size=True,
            marker=marker,
            size=size,
            alpha=alpha,
        )
        present.add(name)

    if x_mode == "params":
        _set_params_axis(ax, xs)
    else:
        _set_date_axis(ax)
    if ylabel_x is None:
        ylabel_x = DATE_YLABEL_X if x_mode == "date" else None
    finish_square_axes(
        ax,
        ylabel,
        ylim_with_references(
            FLOOR_YLIM if ylim is None else ylim, y_rand, y_uff, y_gfnff, y_mmff,
        ),
        ylabel_x=ylabel_x,
    )
    if x_mode == "date" and len(plotted) >= 3:
        fit_x = [p[1] for p in plotted]
        fit_y = [p[2] for p in plotted]
        slope, _, r2, pval = _ols_r2(fit_x, fit_y)
        print(
            f"release date  n={len(plotted)}  slope={slope:+.3f} τ/year  "
            f"R²={r2:.3f}  p={_fmt_p(pval)}"
        )
    elif x_mode == "params" and len(plotted) >= 3:

        fit_x = [p[1] for p in plotted]
        fit_y = [p[2] for p in plotted]
        slope, r2, pval = _draw_trend(ax, fit_x, fit_y, linestyle="-")
        print(
            f"params  n={len(plotted)}  slope={slope:+.3f} τ/decade  "
            f"R²={r2:.3f}  p={_fmt_p(pval)}"
        )
    return present


def write_square(
    out: Path,
    models: list[dict],
    rand: dict,
    uff: dict | None,
    gfnff: dict | None,
    mmff: dict | None,
    dates: dict[str, datetime],
    *,
    y_key: str = "tau",
    ylabel: str = "Conformers Ranking Accuracy (|ΔE| ≥ 2.0 kcal/mol)",
    ylim: tuple[float, float] | None = None,
    show_ci: bool = False,
    x_mode: str = "date",
    marker_by: str = "reasoning",
    size_by_params: bool = False,
    only: str | None = None,
    legend_names: set[str] | None = None,
    figsize: tuple[float, float] | None = None,
    ylabel_x: float | None = None,
) -> None:
    date_layout = figsize is None and x_mode == "date"
    if figsize is None:
        figsize = FIGSIZE_DATE if date_layout else FIGSIZE
    fig, ax, legend_ax = open_square_figure(
        figsize=figsize,
        axes_rect=DATE_AXES if date_layout else SQUARE_AXES,
        legend_rect=DATE_LEGEND if date_layout else SQUARE_LEGEND,
    )
    present = plot_square(
        ax, models, rand, uff, gfnff, mmff, dates,
        y_key=y_key, ylabel=ylabel, ylim=ylim, show_ci=show_ci,
        x_mode=x_mode, marker_by=marker_by, size_by_params=size_by_params,
        only=only, ylabel_x=ylabel_x,
    )
    names = legend_names if legend_names is not None else present
    place_square_legend(
        legend_ax,
        names,
        has_uff=False,
        has_gfnff=False,
        marker_by="reasoning",
        size_by_params=False,
        anchor_bottom=date_layout,
    )
    save_square_figure(fig, out)


def collect_and_write(
    *,
    x_mode: str = "date",
    floors: list[float] | None = None,
    baseline: str = "r2scan3c",
    show_ci: bool = True,
    forcefields: bool = True,
    three_seeds: bool = True,
    bundle_dir: Path | None = None,
    answers_dir: Path | None = None,
    conformers_dir: Path | None = None,
    out: Path | None = None,
    output_tag: str | None = None,
    model_files: dict[str, str] | None = None,
    deltae_json: Path | None = None,
    marker_by: str = "reasoning",
    size_by_params: bool = False,
    only: str | None = None,
) -> list[Path]:
    if bundle_dir is None:
        bundle_dir = geom_seed_dirs()[0][1]
    if conformers_dir is None:
        conformers_dir = geom_conformers_dir()
    if floors is None:
        floors = [2.0]
    floors = list(dict.fromkeys(floors))

    if out is None:
        stem = "tau_vs_params_square" if x_mode == "params" else "tau_vs_release_square"
        if show_ci:
            stem += "_w95"
        if size_by_params:
            stem += "_params"
        if only:
            stem += f"_{only}"
        if output_tag:
            stem += f"_{output_tag}"
        out = Path("figures/release_scatter") / baseline / f"{stem}.png"

    dates = release_dates_by_display()
    override = load_deltae_override(resolve_deltae_json(baseline, deltae_json))
    if model_files is None:
        model_files = dict(
            complete_geom_models(deltae_override=override, warn_unregistered=False)
        )
        if not model_files:
            raise SystemExit(f"No models with 3 complete seeds for {baseline}.")

    models, rand, uff, gfnff, mmff = collect_model_rows(
        bundle_dir,
        conformers_dir,
        answers_dir,
        override,
        three_seeds=three_seeds,
        forcefields="all" if forcefields else "none",
        floors=floors,
        model_files=model_files,
    )

    if output_tag:
        display = display_for_output_tag(output_tag)
        models = [row for row in models if row["name"] == display]
        if not models:
            raise SystemExit(
                f"No complete data available for --output-tag {output_tag!r}."
            )

    written: list[Path] = []
    if floors:
        for floor in floors:
            y_key = f"tau_f{floor}"
            ylabel = f"Conformers Ranking Accuracy (|ΔE| ≥ {floor:.1f} kcal/mol)"
            path = with_floor_stem(out, floor)
            write_square(
                path, models, rand, uff, gfnff, mmff, dates,
                y_key=y_key, ylabel=ylabel, ylim=FLOOR_YLIM,
                show_ci=show_ci,
                x_mode=x_mode,
                marker_by=marker_by,
                size_by_params=size_by_params,
                only=only,
            )
            written.append(path)
    else:
        write_square(
            out,
            models,
            rand,
            uff,
            gfnff,
            mmff,
            dates,
            show_ci=show_ci,
            x_mode=x_mode,
            marker_by=marker_by,
            size_by_params=size_by_params,
            only=only,
        )
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--answers-dir", type=Path, default=None,
                    help="Optional explicit answers dir (default: the seed bundles)")
    ap.add_argument("--conformers-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--output-tag", type=str, default=None)
    ap.add_argument("--baseline", choices=baseline_ids(), default="r2scan3c")
    ap.add_argument(
        "--three-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--model-files", type=str, default=None)
    ap.add_argument(
        "--forcefields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--show-ci",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--deltae-json", type=Path, default=None)
    ap.add_argument("--floors", type=float, nargs="+", default=[2.0], metavar="KCAL")
    ap.add_argument("--x", choices=("date", "params"), default="date")
    ap.add_argument("--marker-by", choices=("reasoning", "arch"), default="reasoning")
    ap.add_argument("--size-by-params", action="store_true")
    ap.add_argument("--only", choices=("os", "closed"), default=None)
    args = ap.parse_args()
    collect_and_write(
        x_mode=args.x,
        floors=list(args.floors),
        baseline=args.baseline,
        show_ci=args.show_ci,
        forcefields=args.forcefields,
        three_seeds=args.three_seeds,
        bundle_dir=args.bundle_dir,
        answers_dir=args.answers_dir,
        conformers_dir=args.conformers_dir,
        out=args.out,
        output_tag=args.output_tag,
        model_files=json.loads(args.model_files) if args.model_files else None,
        deltae_json=args.deltae_json,
        marker_by=args.marker_by,
        size_by_params=args.size_by_params,
        only=args.only,
    )


if __name__ == "__main__":
    main()
