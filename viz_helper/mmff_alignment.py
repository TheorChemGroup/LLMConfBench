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


def plot_scatter_all(a: pd.DataFrame, out: Path) -> Path:
    """MMFF94 vs r2SCAN-3c facets for all 27 molecules (R^2 >= 0.8 green)."""
    u = load_pairs().drop_duplicates(KEYS).copy()
    u["MMFF"] = u[TERMS].sum(axis=1)
    p = u.merge(a[["mol", "b", "R2"]], left_on="molecule_number", right_on="mol")
    aligned = set(a[a.aligned].mol)
    mols = sorted(a.mol)

    fig, axes = plt.subplots(5, 6, figsize=(183 / 25.4, 168 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.075, top=0.965,
                        wspace=0.40, hspace=0.60)
    for ax, mol in zip(axes.ravel(), mols):
        q = p[p.mol == mol]
        ok = mol in aligned
        col = OK if ok else GREY
        ax.scatter(q.MMFF, q.dE_signed, s=3, color=col, alpha=0.5, lw=0,
                   zorder=3)
        lo = float(min(q.MMFF.min(), q.dE_signed.min()))
        hi = float(max(q.MMFF.max(), q.dE_signed.max()))
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        xs = np.array([lo - pad, hi + pad])
        ax.plot(xs, float(q.b.iloc[0]) * xs,
                color=TEXT_COLOR if ok else "#9e9e9e", lw=0.8, zorder=4)
        ax.set_xlim(xs)
        ax.set_ylim(xs)
        ax.locator_params(nbins=3)
        ax.set_title(f"mol {mol}   $R^2$ = {q.R2.iloc[0]:.2f}",
                     fontsize=6.5, color=OK if ok else "#777777", pad=2)
        style_ax(ax)
        ax.tick_params(labelsize=6.0, length=1.8, width=0.5)
    for ax in axes.ravel()[len(mols):]:
        ax.axis("off")

    fig.supxlabel("MMFF94 $\\Delta E$ (kcal mol$^{-1}$)",
                  fontsize=FONTSIZE_LABEL, color="black", y=0.012)
    fig.supylabel("r$^2$SCAN-3c $\\Delta E$ (kcal mol$^{-1}$)",
                  fontsize=FONTSIZE_LABEL, color="black", x=0.008)
    return save_fig(fig, out, dpi=SAVE_DPI)


def main() -> list[Path]:
    a = alignment_table()
    tsv = OUT / "mmff94_alignment_per_molecule.tsv"
    a.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
    return [
        tsv,
        plot_scatter_all(a, OUT / "ed_mmff94_alignment_scatter_all.png"),
    ]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
