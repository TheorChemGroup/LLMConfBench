"""Fitted MMFF94 term contributions to CRA, split by H-bond class (Fig. 1c).

Coefficients from a per-LLM logistic fit on the eight MMFF94-aligned molecules
are scaled by mean term amplitudes on the 19 held-out molecules and normalised
to shares. Depends on ``figures/mmff94_alignment_per_molecule.tsv`` written by
``mmff_alignment.py``.

    python viz_helper/term_contributions.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(Path(__file__).parent)]

from mmff_alignment import KEYS, R2_MIN, TERMS  # noqa: E402
from viz.plot_style import (  # noqa: E402
    ERROR_COLOR,
    FONTSIZE,
    FONTSIZE_LABEL,
    ONE_HALF_COLUMN_WIDTH,
    SAVE_DPI,
    TEXT_COLOR,
    plt,
    style_ax,
)

SHORT = ["Bond", "Angle", "StrBnd", "Torsion", "OOP", "VDW", "Elec"]
LABELS = {"Elec": "Electro-\nstatics", "VDW": "VdW", "Angle": "Angle",
          "StrBnd": "StrBnd", "Torsion": "Torsion", "OOP": "Out-of-\nplane",
          "Bond": "Bond"}
BLUE = "#4a7fb5"
ORANGE = "#e07b39"
COLS = ["x_" + t for t in TERMS]


def load_pairs() -> pd.DataFrame:
    """Pair rows with oriented term differences and H-bond class."""
    terms = pd.read_csv(REPO / "mmff94_terms_by_pair.csv")
    d = pd.read_csv(REPO / "pair_runs_reliable.csv").merge(terms, on=KEYS)
    d["s"] = np.sign(d.dE_signed)
    for t in TERMS:
        d["x_" + t] = d[t] * d.s
    u = d.drop_duplicates(KEYS).copy()
    u["cls"] = np.where(
        u.hb1 == u.hb2, np.where(u.hb1 == 0, "noHB", "bothHB"),
        np.where(((u.hb1 > u.hb2) & (u.dE_signed > 0))
                 | ((u.hb2 > u.hb1) & (u.dE_signed < 0)), "fav", "dec"))
    return d.merge(u[KEYS + ["cls"]], on=KEYS)


def shares(q_all: pd.DataFrame, fit: list[int], test: list[int]) -> pd.DataFrame:
    """Per-LLM normalised term shares for one H-bond class."""
    unique = q_all.drop_duplicates(KEYS)
    amp = (unique[unique.molecule_number.isin(test)]
           .groupby("molecule_number")[COLS]
           .apply(lambda g: g.abs().mean()).mean())
    rows = {}
    for llm, g in q_all[q_all.molecule_number.isin(fit)].groupby("LLM"):
        mols = sorted(g.molecule_number.unique())
        X = g[COLS].values
        M = np.column_stack([(g.molecule_number == m).astype(float) for m in mols])
        y = g.CRA.values
        k = M.shape[1]

        def fg(th):
            b, c = th[:k], th[k:]
            z = M @ b + X @ c
            p = expit(z)
            r = (p - y) / len(y)
            return (np.mean(np.logaddexp(0, z) - y * z) + 1e-3 * np.sum(c * c),
                    np.r_[M.T @ r, X.T @ r + 2e-3 * c])

        th = minimize(fg, np.r_[np.zeros(k), np.ones(7)], jac=True,
                      bounds=[(None, None)] * k + [(0, None)] * 7,
                      method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10}).x
        C = th[k:] * amp.values
        rows[llm] = C / C.sum() if C.sum() > 0 else C * np.nan
    return pd.DataFrame(rows, index=SHORT).T[SHORT]


def plot(S: dict[str, pd.DataFrame], n_no: int, n_any: int, out: Path) -> Path:
    pooled = pd.concat(list(S.values()), axis=0)
    order = sorted(LABELS, key=lambda t: pooled[t].mean(), reverse=True)

    fig = plt.figure(figsize=(ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH))
    ax = fig.add_axes([0.24, 0.355, 0.62, 0.62])
    x = np.arange(len(order))
    for offset, (name, color) in zip((-0.17, 0.17),
                                     (("no H-bond", BLUE), ("any H-bond", ORANGE))):
        bp = ax.boxplot([S[name][t].values for t in order],
                        positions=x + offset, widths=0.30, patch_artist=True,
                        showfliers=False, whis=1.5,
                        medianprops=dict(color=TEXT_COLOR, lw=0.9),
                        whiskerprops=dict(color=ERROR_COLOR, lw=0.6),
                        capprops=dict(color=ERROR_COLOR, lw=0.6),
                        boxprops=dict(lw=0.6, edgecolor=TEXT_COLOR))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.95)
        rng = np.random.default_rng(0)
        for i, t in enumerate(order):
            ax.scatter(i + offset + rng.uniform(-0.055, 0.055, len(S[name])),
                       S[name][t].values, s=4, color=ERROR_COLOR, alpha=0.28,
                       lw=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in order], fontsize=7, rotation=0,
                       linespacing=1.05)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.1, 1.0)
    ax.set_yticks(np.arange(-0.1, 1.01, 0.1))
    ax.set_ylabel("MMFF94 term contributions to CRA prediction",
                  fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor=TEXT_COLOR, lw=0.6,
                             label=f"no H-bond (n = {n_no})"),
                       Patch(facecolor=ORANGE, edgecolor=TEXT_COLOR, lw=0.6,
                             label=f"any H-bond (n = {n_any})"),
                       Line2D([], [], marker="o", ls="", markersize=2.2,
                              markeredgewidth=0, color=ERROR_COLOR, alpha=0.28,
                              label=f"individual LLM (n = {len(S['no H-bond'])})")],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.06,
              fontsize=6.5, frameon=False, borderpad=0.1, labelspacing=0.25,
              handlelength=0.9, handletextpad=0.35)
    style_ax(ax)
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    return out


def main() -> list[Path]:
    a = pd.read_csv(OUT / "mmff94_alignment_per_molecule.tsv", sep="\t")
    fit = sorted(a[a.R2 >= R2_MIN].mol.tolist())
    test = sorted(set(a.mol) - set(fit))
    d = load_pairs()

    S = {
        "no H-bond": shares(d[d.cls == "noHB"], fit, test),
        "any H-bond": shares(d[d.cls != "noHB"], fit, test),
    }
    tsv = OUT / "term_contributions_nohb_anyhb.tsv"
    out = []
    for name, s in S.items():
        t = s.reset_index().rename(columns={"index": "model"})
        t.insert(1, "class", name.replace(" ", "_"))
        out.append(t.melt(id_vars=["model", "class"], var_name="term",
                          value_name="share"))
    pd.concat(out).to_csv(tsv, sep="\t", index=False, float_format="%.6f")

    n_no = d[d.cls == "noHB"].drop_duplicates(KEYS).shape[0]
    n_any = d[d.cls != "noHB"].drop_duplicates(KEYS).shape[0]
    return [tsv, plot(S, n_no, n_any, OUT / "fig1c_term_contributions.png")]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
