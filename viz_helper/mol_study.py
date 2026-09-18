"""Concreteness study: tau vs relative abundance of H-bond / energy mentions.

Python port of the R/ggplot script (tidyverse + geom_smooth loess, span=1.5):

  * X = (n_kcal + n_hbond) / words        (relative abundance per explanation)
  * 3-sigma per-model outlier filter on X, then X / mean(X)  -> relative to the
    model's own average
  * Y = tau, one LOESS curve per model, coloured by the model's mean tau
    (red = low ... green = high, limits -1..1)

Figures (saved at dpi=2000 in figures/r2scan3c/):
  AllLoesses.png  - all model curves overlaid in one panel (no points)
  BestModels.png  - GPT-6 Astra + Claude Opus 5, faceted, labelled points
  FullPlot.png    - 28-panel facet grid, labelled points + smooth + CI band

    python viz_helper/mol_study.py
    python viz_helper/mol_study.py --data "LLM as confsearch - Tau vs. Concreteness.csv"
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.colors import LinearSegmentedColormap, Normalize

from config.models_config import TAU_FLOOR
from viz.plot_style import (
    ONE_HALF_COLUMN_WIDTH,
    plt,
    save_fig,
    style_ax,
)

DEFAULT_DATA = _REPO / "figures" / "r2scan3c" / "concreteness_vs_tau.tsv"
DEFAULT_OUTDIR = _REPO / "figures" / "r2scan3c"
BEST_MODELS = ("GPT-6 Astra", "GPT-5")
XLABEL = "Relative abundance of H-bond mentions"
YTICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
FLOOR = TAU_FLOOR
YLABEL = f"Conformers Ranking Accuracy (|ΔE| ≥ {FLOOR:.1f} kcal/mol)"
SPAN = 1.5
CMAP = LinearSegmentedColormap.from_list(
    "mt", ["#c62828", "#ef6c00", "#9e9e9e", "#4fc3f7", "#2e7d32"]
)
NORM = Normalize(vmin=-1.0, vmax=1.0)


def load_prepared(path: Path) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Return (d4, model order by mean tau, mean tau per model)."""
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    d = pd.read_csv(path, sep=sep)
    d = d[d["tau"].notna()].copy()
    d["X"] = d["n_hbond"] / d["words"]
    d = d[d["X"].notna()].copy()

    mt = d.groupby("model")["tau"].mean()
    order = list(mt.sort_values(ascending=False).index)


    sd = d.groupby("model")["X"].transform("std")
    mean = d.groupby("model")["X"].transform("mean")
    fact = (d["X"] - mean).abs() / sd
    d = d[(fact < 3) | sd.isna() | (sd == 0)]
    d["X"] = d.groupby("model")["X"].transform(lambda s: s / s.mean())
    d["mt"] = d["model"].map(mt)
    return d, order, mt


def _color(mt: float):
    return CMAP(NORM(float(np.clip(mt, -1.0, 1.0))))


def loess(
    x,
    y,
    x_eval,
    *,
    span: float = SPAN,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Local quadratic regression with tricube weights.

    Mirrors R's `loess(span = 1.5, degree = 2)`: q = ceil(span * n) nearest
    neighbours (clipped to n, so span > 1 covers all points), weights
    (1 - (d/dmax)^3)^3.  Returns (fit, 1.96*se) at ``x_eval``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    n = len(x)
    q = min(n, max(degree + 1, math.ceil(span * n)))

    fit, half = [], []
    for x0 in x_eval:
        dist = np.abs(x - x0)
        idx = np.argpartition(dist, q - 1)[:q]
        dmax = dist[idx].max()
        if dmax <= 0:
            dmax = 1e-9
        u = np.clip(dist[idx] / dmax, 0, 1)
        w = (1 - u**3) ** 3

        A = np.vander(x[idx] - x0, degree + 1, increasing=True)
        Aw = A * np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(Aw, y[idx] * np.sqrt(w), rcond=None)
        fit.append(beta[0])

        M = np.linalg.inv(Aw.T @ Aw)
        dof = max(1, q - (degree + 1))
        sigma2 = float(np.sum(w * (y[idx] - A @ beta) ** 2) / dof)
        half.append(1.96 * math.sqrt(max(0.0, M[0, 0] * sigma2)))
    return np.array(fit), np.array(half)


def _curve(g: pd.DataFrame, npts: int = 100):
    xe = np.linspace(g["X"].min(), g["X"].max(), npts)
    f, se = loess(g["X"].values, g["tau"].values, xe)
    return xe, f, se


def _labelled_panel(ax, g: pd.DataFrame, fontsize: float) -> None:
    col = _color(g["mt"].iloc[0])
    for _, r in g.iterrows():
        ax.text(r["X"], r["tau"], str(int(r["label"])), color=col,
                fontsize=fontsize, ha="center", va="center", alpha=0.85)
    if len(g) >= 4:
        xe, f, se = _curve(g)
        ax.fill_between(xe, f - se, f + se, color=col, alpha=0.15, lw=0)
        ax.plot(xe, f, color=col, lw=1.0)


def plot_full(d, order, outdir: Path) -> Path:
    n = len(order)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(18 / 2.54, 24 / 2.54))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, order):
        g = d[d["model"] == model]
        _labelled_panel(ax, g, fontsize=3)
        style_ax(ax, square=False)
        ax.tick_params(labelsize=4)
        ax.set_title(model, fontsize=6)
        ax.set_yticks(YTICKS)
        ax.set_ylim(-1, 1)
    for ax in axes[n:]:
        ax.axis("off")
    fig.supxlabel(XLABEL, fontsize=7)
    fig.supylabel(YLABEL, fontsize=7)
    fig.tight_layout()
    return save_fig(fig, outdir / "FullPlot.png")


def plot_combined(d, order, outdir: Path, outname: str = "ConcretenessTau.png") -> Path:
    """All-models curves + the two best models in one row (no panel letters)."""
    fig, axes = plt.subplots(1, 3, figsize=(3 * ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH),
                             sharey=True, gridspec_kw={"wspace": 0.06})
    ax_a, ax_b, ax_all = axes

    for ax, model in zip((ax_a, ax_b), BEST_MODELS):
        _labelled_panel(ax, d[d["model"] == model], fontsize=8)
        ax.set_title(model, fontsize=12)

    for model in order:
        g = d[d["model"] == model]
        if len(g) < 4:
            continue
        xe, f, _ = _curve(g)
        ax_all.plot(xe, f, color=_color(g["mt"].iloc[0]), lw=1.0)
    ax_all.set_title("All models", fontsize=12)

    for ax in axes:
        style_ax(ax, square=False)
        ax.tick_params(labelsize=9)
        ax.set_yticks(YTICKS)
        ax.set_ylim(-1.06, 1.0)
        ax.set_xlabel(XLABEL, fontsize=10)
    ax_a.set_ylabel(YLABEL, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_all, orientation="vertical",
                        fraction=0.05, pad=0.02, shrink=0.6)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_label("Model mean τ", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    return save_fig(fig, outdir / outname)


def build_and_save(data: Path = DEFAULT_DATA, outdir: Path = DEFAULT_OUTDIR) -> list[Path]:
    d, order, mt = load_prepared(data)
    print(f"{len(d)} points from {d['model'].nunique()} models  (data: {data})")
    print("model order by mean tau:", ", ".join(f"{m} ({mt[m]:+.2f})" for m in order[:5]), "...")
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        plot_combined(d, order, outdir),
        plot_full(d, order, outdir),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="CSV/TSV with model,tau,n_kcal,n_hbond,words,label,seed")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    build_and_save(args.data, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
