"""MMFF94 vs r2SCAN-3c agreement per molecule (Extended Data Fig. S9).

For every reliable pair, MMFF94 total energy differences are compared with the
r2SCAN-3c reference through the origin. Eight molecules with R^2 >= 0.8 enter
the term-contribution fit (``term_contributions.py``).

    python viz_helper/mmff_alignment.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz")]

from viz.plot_style import (  # noqa: E402
    FONTSIZE,
    FONTSIZE_LABEL,
    SAVE_DPI,
    TEXT_COLOR,
    plt,
    save_fig,
    style_ax,
)

TERMS = ["dE_Bond", "dE_Angle", "dE_StrBnd", "dE_Torsion", "dE_OOP",
         "dE_VDW", "dE_Electrostatic"]
KEYS = ["molecule_number", "conf1", "conf2"]
R2_MIN = 0.8
COLOR = "#4a7fb5"
OK = "#2e7d32"
GREY = "#bdbdbd"


def load_pairs() -> pd.DataFrame:
    """Pair rows with MMFF94 term differences (one row per pair and run)."""
    terms = pd.read_csv(REPO / "mmff94_terms_by_pair.csv")
    return pd.read_csv(REPO / "pair_runs_reliable.csv").merge(terms, on=KEYS)


def alignment_table() -> pd.DataFrame:
    """Through-origin R^2 and sign agreement per molecule."""
    u = load_pairs().drop_duplicates(KEYS).copy()
    u["MMFF"] = u[TERMS].sum(axis=1)
    rows = []
    for mol, g in u.groupby("molecule_number"):
        x, y = g.MMFF.values, g.dE_signed.values
        b = float(x @ y / (x @ x))
        r2 = 1 - float(((y - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        rows.append({"mol": int(mol), "b": b, "R2": r2,
                     "sign": float((np.sign(y) == np.sign(b * x)).mean())})
    a = pd.DataFrame(rows).sort_values("mol").reset_index(drop=True)
    a["aligned"] = a.R2 >= R2_MIN
    return a


def plot_r2(a: pd.DataFrame, out: Path) -> Path:
    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    ax = fig.add_axes([0.105, 0.30, 0.88, 0.62])
    cols = [OK if ok else GREY for ok in a.aligned]
    ax.bar(a.mol.astype(str), a.R2, color=cols, width=0.75, lw=0)
    ax.axhline(0, color="#888888", lw=0.6, zorder=1)
    ax.axhline(R2_MIN, color=TEXT_COLOR, lw=0.9, ls="--", zorder=2)
    ax.text(1.0, 1.025, r"dashed: $R^2 = 0.8$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FONTSIZE, color=TEXT_COLOR)
    ax.set_xlabel("Molecule", fontsize=FONTSIZE_LABEL, color=TEXT_COLOR)
    ax.set_ylabel(r"$R^2$ (through origin)", fontsize=FONTSIZE_LABEL,
                  color=TEXT_COLOR)
    ax.set_ylim(-3.05, 1.05)
    ax.set_yticks(np.arange(-3, 1.01, 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.7, 26.7)
    style_ax(ax)
    return save_fig(fig, out, dpi=SAVE_DPI)


def plot_scatter(a: pd.DataFrame, out: Path) -> Path:
    fit = sorted(a[a.aligned].mol.tolist())
    u = load_pairs().drop_duplicates(KEYS).copy()
    u["MMFF"] = u[TERMS].sum(axis=1)
    p = u[u.molecule_number.isin(fit)].merge(a[["mol", "b", "R2"]],
                                             left_on="molecule_number",
                                             right_on="mol")
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 96 / 25.4))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.135, top=0.93,
                        wspace=0.30, hspace=0.42)
    for ax, mol in zip(axes.ravel(), fit):
        q = p[p.mol == mol]
        ax.scatter(q.MMFF, q.dE_signed, s=4, color=COLOR, alpha=0.45, lw=0,
                   zorder=3)
        lo = float(min(q.MMFF.min(), q.dE_signed.min()))
        hi = float(max(q.MMFF.max(), q.dE_signed.max()))
        pad = 0.06 * (hi - lo)
        xs = np.array([lo - pad, hi + pad])
        ax.plot(xs, float(q.b.iloc[0]) * xs, color=TEXT_COLOR, lw=0.9, zorder=4)
        ax.set_xlim(xs)
        ax.set_ylim(xs)
        ax.set_title(f"mol {mol}   $R^2$ = {q.R2.iloc[0]:.2f}",
                     fontsize=FONTSIZE, color=TEXT_COLOR, pad=2)
        ax.tick_params(labelsize=7)
        style_ax(ax)
    fig.supxlabel("MMFF94 $\\Delta E$ (kcal mol$^{-1}$)",
                  fontsize=FONTSIZE_LABEL, color=TEXT_COLOR, y=0.015)
    fig.supylabel("r$^2$SCAN-3c $\\Delta E$ (kcal mol$^{-1}$)",
                  fontsize=FONTSIZE_LABEL, color=TEXT_COLOR, x=0.012)
    return save_fig(fig, out, dpi=SAVE_DPI)


def main() -> list[Path]:
    a = alignment_table()
    tsv = OUT / "mmff94_alignment_per_molecule.tsv"
    a.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
    return [
        tsv,
        plot_r2(a, OUT / "ed_mmff94_alignment_r2.png"),
        plot_scatter(a, OUT / "ed_mmff94_alignment_scatter.png"),
    ]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
