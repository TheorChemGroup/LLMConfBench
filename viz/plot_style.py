from __future__ import annotations

from pathlib import Path

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

FONTSIZE = 9
FONTSIZE_LABEL = 10
MM_PER_INCH = 25.4
ONE_HALF_COLUMN_WIDTH = 127.0 / MM_PER_INCH
DOUBLE_COLUMN_WIDTH = 183.0 / MM_PER_INCH
ERROR_COLOR = "#3F3F3F"
TEXT_COLOR = "#333333"
REF_RANDOM = "#9ca3af"
REF_UFF = "#D9CB00"
REF_GFNFF = "#00E900"
REF_MMFF = "#00003C"
REF_MMFF_LINESTYLE = (0, (3, 1, 1, 1))
FIGSIZE_SQUARE = (ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH)
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


def save_fig(fig: plt.Figure, path, *, dpi: int = SAVE_DPI) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out}")
    return out


__all__ = [
    "DOUBLE_COLUMN_WIDTH",
    "ERROR_COLOR",
    "FIGSIZE_SQUARE",
    "FONTSIZE",
    "FONTSIZE_LABEL",
    "ONE_HALF_COLUMN_WIDTH",
    "REF_GFNFF",
    "REF_MMFF",
    "REF_MMFF_LINESTYLE",
    "REF_RANDOM",
    "REF_UFF",
    "legend_below",
    "plt",
    "save_fig",
    "style_ax",
]
