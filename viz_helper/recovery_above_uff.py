"""Recovery of the MMFF94 terms by the models that beat UFF.

One logistic fit per model on the eight MMFF94-aligned molecules, all pairs at
once (no H-bond split), lambda = 0.01, coefficients normalised to sum to 1.
Only models whose mean-over-molecules CRA on the reliable pairs exceeds UFF on
the same pairs are shown. Numbers under the axis are the mean absolute change of
each term within the 1,391 pairs of those eight molecules.

    python viz_helper/recovery_above_uff.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(Path(__file__).parent)]

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from decoy_common import FLOOR, bench_molecules, load_r2scan  # noqa: E402
from term_contributions import (  # noqa: E402
    LABELS,
    SHORT,
    amplitudes,
    fit_terms,
    load_pairs,
    split_molecules,
)
from viz.plot_style import (  # noqa: E402
    ERROR_COLOR,
    FONTSIZE,
    FONTSIZE_LABEL,
    FIGSIZE_SQUARE,
    SAVE_DPI,
    TEXT_COLOR,
    plt,
    style_ax,
)

LAM = 0.01
COLOUR = "#7e57c2"
UFF_CACHE = REPO / "geom-qm9/output_30_confs/.ff_cache/uff_smiles"
ORDER = ["Elec", "VDW", "StrBnd", "Angle", "Bond", "Torsion", "OOP"]
TERM_INDEX = {term: i for i, term in enumerate(SHORT)}
HIGHLIGHT = (("GPT-6 Astra", "*", 34), ("Claude Opus 5", "s", 11),
             ("GPT-5", "^", 13))


def uff_macro_cra() -> float:
    """UFF correct-rank share, averaged over molecules, on the reliable pairs."""
    energy = load_r2scan()
    per_mol = []
    for folder in bench_molecules():
        rank = json.loads((UFF_CACHE / f"{folder}.json").read_text())["ranking"]
        pos = {int(c): i for i, c in enumerate(rank)}
        de = {int(k): float(v) for k, v in energy[folder].items()}
        hits = total = 0
        for c1, c2 in combinations(range(30), 2):
            d_e = de[c2] - de[c1]
            if abs(d_e) < FLOOR:
                continue
            lower = c1 if pos[c1] < pos[c2] else c2
            hits += int(lower == (c1 if d_e > 0 else c2))
            total += 1
        per_mol.append(hits / total)
    return float(np.mean(per_mol))


def models_above_uff(d: pd.DataFrame) -> list[str]:
    """Models whose mean-over-molecules CRA exceeds UFF's on the same pairs."""
    bar = uff_macro_cra()
    macro = d.groupby(["LLM", "molecule_number"]).CRA.mean().groupby("LLM").mean()
    keep = sorted(macro[macro > bar].index)
    print(f"UFF macro CRA {bar:.3f}; {len(keep)} of {len(macro)} models above")
    return keep


def collect(lam: float = LAM) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    """Normalised coefficients per model, term amplitudes, pair counts."""
    d = load_pairs()
    fit, _held = split_molecules()
    fitted = d[d.molecule_number.isin(fit)]
    keep = models_above_uff(d)

    coef = pd.DataFrame(
        {llm: fit_terms(g, lam) for llm, g in fitted.groupby("LLM")
         if llm in keep}, index=SHORT).T[SHORT]
    coef = coef.div(coef.sum(axis=1), axis=0)
    amp = amplitudes(d, set(fit))
    n_pairs = len(fitted.drop_duplicates(["molecule_number", "conf1", "conf2"]))
    return coef, amp, n_pairs, len(keep)


def plot(coef: pd.DataFrame, amp: np.ndarray, n_pairs: int, out: Path) -> Path:
    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = fig.add_axes([0.24, 0.355, 0.62, 0.62])
    x = np.arange(len(ORDER))
    rng = np.random.default_rng(0)
    small = FONTSIZE_LABEL - 2

    vals = {t: coef[t].values for t in ORDER}
    bp = ax.boxplot([vals[t] for t in ORDER], positions=x, widths=0.42,
                    patch_artist=True, showfliers=False, whis=1.5,
                    medianprops=dict(color=TEXT_COLOR, lw=0.9),
                    whiskerprops=dict(color=ERROR_COLOR, lw=0.6),
                    capprops=dict(color=ERROR_COLOR, lw=0.6),
                    boxprops=dict(lw=0.6, edgecolor=TEXT_COLOR))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOUR)
        patch.set_alpha(0.95)
    for i, t in enumerate(ORDER):
        ax.scatter(i + rng.uniform(-0.085, 0.085, len(vals[t])), vals[t], s=4,
                   color=ERROR_COLOR, alpha=0.28, lw=0, zorder=3)
        for llm, marker, size in HIGHLIGHT:
            if llm in coef.index:
                ax.scatter(i, coef.loc[llm, t], s=size, marker=marker,
                           facecolor=COLOUR, edgecolor="black", linewidth=0.45,
                           zorder=5)
        ax.text(i, -0.15, f"{amp[TERM_INDEX[t]]:.2f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=small, color=COLOUR)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in ORDER], fontsize=5.6, rotation=0,
                       linespacing=1.05, color=TEXT_COLOR)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    top = float(max(v.max() for v in vals.values())) * 1.06
    ax.set_ylim(-0.03, top)
    ax.set_yticks(np.arange(0, top, 0.5))
    ax.set_ylabel("Recovery of the energetic term by LLMs\n"
                  "normalised to sum to 1",
                  fontsize=FONTSIZE_LABEL, linespacing=1.25)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.legend(handles=[Patch(facecolor=COLOUR, edgecolor=TEXT_COLOR, lw=0.6,
                             label="8 MMFF94 molecules fit"),
                       Line2D([], [], marker="o", ls="", markersize=2.2,
                              markeredgewidth=0, color=ERROR_COLOR, alpha=0.28,
                              label=f"LLM above UFF (n = {len(coef)})"),
                       *[Line2D([], [], marker=marker, ls="", markersize=3.4,
                                markerfacecolor="#cfcfcf", markeredgecolor="black",
                                markeredgewidth=0.45, label=llm)
                         for llm, marker, _size in HIGHLIGHT]],
              loc="upper right", borderaxespad=0.06, fontsize=6.5, frameon=False,
              borderpad=0.1, labelspacing=0.25, handlelength=0.9,
              handletextpad=0.35)
    style_ax(ax)
    fig.text(0.55, 0.238,
             "Mean absolute changes of the terms within "
             f"the {n_pairs:,} pairs\nof these molecules, kcal/mol",
             ha="center", va="top", fontsize=small, color=TEXT_COLOR,
             linespacing=1.45)
    fig.text(0.55, 0.125, "MMFF94 terms", ha="center", fontsize=FONTSIZE_LABEL)
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    return out


def main() -> Path:
    coef, amp, n_pairs, _n_models = collect()
    return plot(coef, amp, n_pairs,
                OUT / "recovery_pooled_fit8_normalised_above_uff.png")


if __name__ == "__main__":
    print(f"Saved {main()}")
