"""Concreteness panels: H-bond mention abundance vs CRA (Fig. 2d-f).

x: relative abundance of H-bond mentions (per molecule, per seed)
y: CRA across fair (equal-H-bond) pairs for that seed
points labelled by molecule number, LOESS fit with 95 % band.

    python viz_helper/concreteness_panel.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(REPO / "viz_helper")]

from viz.plot_style import (  # noqa: E402
    FONTSIZE,
    FONTSIZE_LABEL,
    ONE_HALF_COLUMN_WIDTH,
    SAVE_DPI,
    plt,
    style_ax,
)
from viz_helper.mol_study import loess  # noqa: E402

COLOR = "#4f8fa6"
MODELS = (("GPT-5", "gpt5"), ("GPT-6 Astra", "gpt6"), ("Claude Opus 5", "opus5"))
YLIM = (-1.05, 1.05)
YTICKS = np.arange(-1.0, 1.01, 0.5)


def cra_for(pairs: pd.DataFrame) -> pd.DataFrame:
    return (2 * pairs.groupby(["LLM", "molecule_number", "run"]).CRA.mean() - 1
            ).rename("cra").reset_index()


def merged(t: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    g = t.merge(cra_for(pairs), left_on=["model", "label", "run"],
                right_on=["LLM", "molecule_number", "run"], how="inner")
    return g.dropna(subset=["abundance", "cra"])


def panel(q: pd.DataFrame, model: str, ylabel: str, path: Path) -> Path:
    fig = plt.figure(figsize=(ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH))
    ax = fig.add_axes([0.24, 0.355, 0.62, 0.60])

    xs = np.linspace(q.abundance.min(), q.abundance.max(), 100)
    f, se = loess(q.abundance.values, q.cra.values, xs)
    ax.fill_between(xs, f - se, f + se, color=COLOR, alpha=0.15, lw=0)
    ax.plot(xs, f, color=COLOR, lw=1.0, zorder=2)

    for r in q.itertuples():
        ax.text(r.abundance, r.cra, f"{r.label}", fontsize=5.5, color=COLOR,
                alpha=0.85, ha="center", va="center", zorder=3)

    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_xlabel("Relative abundance of H-bond mentions",
                  fontsize=FONTSIZE_LABEL, color="black")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL, color="black")
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_title(model, fontsize=FONTSIZE_LABEL, color="black", pad=4)
    style_ax(ax)
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"saved {path} ({len(q)} points)")
    return path


def main() -> list[Path]:
    d = pd.read_csv(REPO / "pair_runs_reliable.csv")
    t = pd.read_csv(REPO / "figures/concreteness_vs_cra_fp.tsv", sep="\t")
    t["X"] = t.n_hbond / t.words
    # Same per-model 3-sigma filter as mol_study.load_prepared (concreteness_full
    # and the H-bond panel of FullPlotConcepts).
    sd = t.groupby("model")["X"].transform("std")
    mean = t.groupby("model")["X"].transform("mean")
    fact = (t.X - mean).abs() / sd
    t = t[(fact < 3) | sd.isna() | (sd == 0)].copy()
    t["abundance"] = t.X / t.groupby("model")["X"].transform("mean")
    t["run"] = t.seed - 41

    g = merged(t, d[d.hb1 == d.hb2])
    print(f"CRA across fair pairs: {len(g)} points, {g.label.nunique()} molecules")
    return [panel(g[g.model == model], model, "CRA across fair pairs",
                  OUT / f"concreteness_{short}.png")
            for model, short in MODELS]


if __name__ == "__main__":
    main()
