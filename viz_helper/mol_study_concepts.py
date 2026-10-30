"""FullPlot-style facet of concept abundance vs ranking accuracy.

One panel per chemical concept, all 28 model LOESS curves overlaid and coloured
by the model's mean CRA. The H-bond panel uses the same rows, filter, X
normalization and LOESS domain as ``FullPlot.png`` (viz_helper/mol_study.py), so
its 28 curves are identical to the curves in that figure.

    python viz_helper/mol_study_concepts.py
"""
from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "viz_helper"))

from concepts import CONCEPTS, TABLE_TSV
from mol_study import (
    CMAP,
    NORM,
    YLABEL,
    YLIM,
    YTICKS,
    _color,
    _curve,
)

from viz.plot_style import plt, save_fig, style_ax

DEFAULT_DATA = TABLE_TSV
DEFAULT_OUT = _REPO / "figures" / "FullPlotConcepts.png"
NCOL = 4
FIGSIZE = (18 / 2.54, 16 / 2.54)


def load_prepared(path: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, pd.Series]:
    """Return (table, mean CRA per model) with tau present."""
    d = pd.read_csv(path, sep="\t")
    d = d[d["tau"].notna()].copy()
    mt = d.groupby("model")["tau"].mean()
    return d, mt.sort_values(ascending=False)


def prepare(d: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Relative abundance X, 3-sigma filter and per-model normalization.

    Mirrors ``mol_study.load_prepared`` for the H-bond column.
    """
    out = d.copy()
    out["X"] = out[concept] / out["words"]
    out = out[out["X"].notna() & np.isfinite(out["X"])].copy()
    sd = out.groupby("model")["X"].transform("std")
    mean = out.groupby("model")["X"].transform("mean")
    fact = (out["X"] - mean).abs() / sd
    out = out[(fact < 3) | sd.isna() | (sd == 0)].copy()
    out["X"] = out.groupby("model")["X"].transform(
        lambda s: s / s.mean() if s.mean() != 0 else np.nan
    )
    out = out[out["X"].notna()].copy()
    return out


def plot_facet(
    d: pd.DataFrame,
    mt: pd.Series,
    *,
    out: Path = DEFAULT_OUT,
) -> Path:
    nrow = math.ceil(len(CONCEPTS) / NCOL)
    fig, axes = plt.subplots(nrow, NCOL, figsize=FIGSIZE, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, concept in zip(axes, CONCEPTS):
        g = prepare(d, concept)
        plotted = 0
        for model in mt.index:
            gm = g[g["model"] == model]
            if len(gm) < 4 or gm["X"].mean() == 0:
                continue
            try:
                xe, fit, _se = _curve(gm)
            except np.linalg.LinAlgError:
                continue
            ax.plot(xe, fit, color=_color(mt[model]), lw=0.8)
            plotted += 1
        style_ax(ax, square=False)
        ax.tick_params(labelsize=5)
        ax.set_yticks(YTICKS)
        ax.set_ylim(*YLIM)
        ax.set_title(textwrap.fill(concept, 22), fontsize=7)
        print(f"  {concept:32s} {plotted} models")

    for ax in axes[len(CONCEPTS):]:
        ax.axis("off")

    fig.supxlabel("Relative abundance of concept mentions", fontsize=8)
    fig.supylabel(YLABEL, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 0.925, 1))

    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cax = fig.add_axes((0.94, 0.32, 0.012, 0.36))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(YTICKS)
    cbar.set_label("Model mean τ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return save_fig(fig, out)


def build_and_save(
    data: Path = DEFAULT_DATA,
    out: Path = DEFAULT_OUT,
) -> Path:
    d, mt = load_prepared(data)
    print(f"{len(d)} points from {d['model'].nunique()} models  (data: {data})")
    return plot_facet(d, mt, out=out)


if __name__ == "__main__":
    build_and_save()
