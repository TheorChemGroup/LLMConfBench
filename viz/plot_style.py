from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 9
FONTSIZE_LABEL = 10
MM_PER_INCH = 25.4
SINGLE_COLUMN_WIDTH = 89.0 / MM_PER_INCH
ONE_HALF_COLUMN_WIDTH = 127.0 / MM_PER_INCH
DOUBLE_COLUMN_WIDTH = 183.0 / MM_PER_INCH
MAX_FIGURE_HEIGHT = 225.0 / MM_PER_INCH
ERROR_COLOR = "#3F3F3F"
STEM_COLOR = "#C4C4C4"
TEXT_COLOR = "#333333"
REF_RANDOM = "#9ca3af"
REF_UFF = "#10b981"
REF_GFNFF = "#38bdf8"
REF_MMFF = "#4d4d4d"
REF_MMFF_LINESTYLE = (0, (3, 1, 1, 1))
FIGSIZE_SQUARE = (ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH)
FIGSIZE_DOUBLE_WIDE = (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH / 2.0)
SAVE_DPI = 500

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE_LABEL,
        "axes.labelsize": FONTSIZE_LABEL,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "figure.titlesize": FONTSIZE_LABEL,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "xtick.top": False,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def style_ax(ax: plt.Axes, *, square: bool = False) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#333333")
    ax.tick_params(
        top=False,
        right=False,
        left=True,
        bottom=True,
        which="both",
        direction="out",
        length=3,
        width=0.6,
        labelsize=FONTSIZE,
        color="#333333",
    )
    ax.set_axisbelow(True)
    if square:
        ax.set_box_aspect(1)


def plot_lollipop(
    ax: plt.Axes,
    y: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    colors: list[str],
    labels: list[str],
    *,
    fmt: str = "{:+.3f}",
    x0: float = 0.0,
) -> None:
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.hlines(y, x0, values, color=STEM_COLOR, linewidth=1.1, zorder=1)
    ax.errorbar(
        values,
        y,
        xerr=errors,
        fmt="none",
        ecolor=ERROR_COLOR,
        elinewidth=0.7,
        capsize=0,
        zorder=2,
    )
    ax.scatter(
        values,
        y,
        s=38,
        c=colors,
        zorder=3,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONTSIZE)
    finite = np.isfinite(values) & np.isfinite(errors)
    if not np.any(finite):
        return
    right = float(np.nanmax(values[finite] + errors[finite]))
    left = min(x0, float(np.nanmin(values[finite] - errors[finite])))
    span = max(right - left, 0.05)
    pad = 0.08 * span
    for yi, val, err in zip(y, values, errors):
        if not np.isfinite(val):
            continue
        e = err if np.isfinite(err) else 0.0
        ax.text(
            val + e + pad,
            yi,
            fmt.format(val),
            va="center",
            ha="left",
            fontsize=FONTSIZE,
            color=TEXT_COLOR,
            clip_on=False,
        )
    ax.set_xlim(left - 0.02 * span, right + 0.38 * span)


def legend_below(obj, *, ncol: int = 4, **kwargs) -> None:
    kw = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.16),
        "ncol": ncol,
        "fontsize": FONTSIZE,
        "frameon": False,
        "columnspacing": 1.2,
        "handletextpad": 0.5,
        "borderaxespad": 0.0,
    }
    kw.update(kwargs)
    obj.legend(**kw)


def save_fig(fig: plt.Figure, path, *, dpi: int = SAVE_DPI) -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out}")


__all__ = [
    "DOUBLE_COLUMN_WIDTH",
    "ERROR_COLOR",
    "FIGSIZE_DOUBLE_WIDE",
    "FIGSIZE_SQUARE",
    "FONTSIZE",
    "FONTSIZE_LABEL",
    "MAX_FIGURE_HEIGHT",
    "ONE_HALF_COLUMN_WIDTH",
    "REF_GFNFF",
    "REF_MMFF",
    "REF_MMFF_LINESTYLE",
    "REF_RANDOM",
    "REF_UFF",
    "SINGLE_COLUMN_WIDTH",
    "legend_below",
    "plot_lollipop",
    "plt",
    "save_fig",
    "style_ax",
]
