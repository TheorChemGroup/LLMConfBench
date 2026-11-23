"""Amplified term contributions (Fig. 1c, unnormalised companion panel).

Coefficients are fitted on the eight MMFF94-aligned molecules and multiplied by
the mean absolute change of each term over all 3,802 pairs, so the height shows
how much a term moves the model's ranking decision. Numbers under the bars give
that mean change in kcal/mol.

    python viz_helper/term_contributions_amplified.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(Path(__file__).parent)]

from term_contributions import (  # noqa: E402
    BLUE,
    LABELS,
    ORANGE,
    SHORT,
    amplitudes,
    class_frames,
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

ORDER = ["Elec", "VDW", "StrBnd", "Angle", "Bond", "Torsion", "OOP"]
TERM_INDEX = {term: i for i, term in enumerate(SHORT)}
CLASSES = (("no H-bond", BLUE, "noHB"), ("any H-bond", ORANGE, "other"))


def collect() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], int]:
    """Amplified contributions and amplitudes per class, plus the pair count."""
    fit, test = split_molecules()
    all_mols = set(fit) | test
    d = load_pairs()
    frames = class_frames(d)

    coef, amps = {}, {}
    for name, _color, key in CLASSES:
        sub = frames[key]
        coef[name] = pd.DataFrame(
            {llm: fit_terms(g)
             for llm, g in sub[sub.molecule_number.isin(fit)].groupby("LLM")},
            index=SHORT).T[SHORT]
        amps[name] = amplitudes(sub, all_mols)
    n_pairs = len(d.drop_duplicates(["molecule_number", "conf1", "conf2"]))
    return coef, amps, n_pairs


def plot(coef: dict[str, pd.DataFrame], amps: dict[str, np.ndarray],
         n_pairs: int, out: Path) -> Path:
    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = fig.add_axes([0.24, 0.355, 0.62, 0.62])
    x = np.arange(len(ORDER))
    rng = np.random.default_rng(0)
    for row_y, (off, (name, color, _key)) in zip(
            (-0.155, -0.215), zip((-0.17, 0.17), CLASSES)):
        df = coef[name]
        vals = {t: df[t].values * amps[name][TERM_INDEX[t]] for t in ORDER}
        bp = ax.boxplot([vals[t] for t in ORDER], positions=x + off, widths=0.30,
                        patch_artist=True, showfliers=False, whis=1.5,
                        medianprops=dict(color=TEXT_COLOR, lw=0.9),
                        whiskerprops=dict(color=ERROR_COLOR, lw=0.6),
                        capprops=dict(color=ERROR_COLOR, lw=0.6),
                        boxprops=dict(lw=0.6, edgecolor=TEXT_COLOR))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.95)
        for i, t in enumerate(ORDER):
            ax.scatter(i + off + rng.uniform(-0.055, 0.055, len(vals[t])),
                       vals[t], s=4, color=ERROR_COLOR, alpha=0.28, lw=0,
                       zorder=3)
            ax.text(i + off, row_y, f"{amps[name][TERM_INDEX[t]]:.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in ORDER], fontsize=5.6, rotation=0,
                       linespacing=1.05, color=TEXT_COLOR)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(-0.15, 4.4)
    ax.set_yticks(np.arange(0, 4.1, 1))
    ax.set_ylabel("Recovery of the energetic term by LLMs\n"
                  "multiplied by its mean absolute change",
                  fontsize=FONTSIZE_LABEL, linespacing=1.25)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor=TEXT_COLOR, lw=0.6,
                             label="no H-bond"),
                       Patch(facecolor=ORANGE, edgecolor=TEXT_COLOR, lw=0.6,
                             label="any H-bond"),
                       Line2D([], [], marker="o", ls="", markersize=2.2,
                              markeredgewidth=0, color=ERROR_COLOR, alpha=0.28,
                              label=f"individual LLM (n = {len(coef['no H-bond'])})")],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.06,
              fontsize=6.5, frameon=False, borderpad=0.1, labelspacing=0.25,
              handlelength=0.9, handletextpad=0.35)
    style_ax(ax)
    fig.text(0.55, 0.163,
             f"Mean absolute changes of the terms within all {n_pairs:,} pairs, kcal/mol",
             ha="center", fontsize=6, color=TEXT_COLOR)
    fig.text(0.55, 0.115, "MMFF94 terms", ha="center", fontsize=FONTSIZE_LABEL)
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    return out


def main() -> list[Path]:
    coef, amps, n_pairs = collect()
    return [plot(coef, amps, n_pairs,
                 OUT / "fig1c_amplified_term_contributions.png")]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
