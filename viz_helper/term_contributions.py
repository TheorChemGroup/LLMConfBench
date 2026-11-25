"""Shared MMFF94 term fit used by the manuscript figures.

Coefficients come from a per-LLM logistic fit on the eight MMFF94-aligned
molecules; ``recovery_above_uff.py`` plots them. Depends on
``figures/mmff94_alignment_per_molecule.tsv`` written by ``mmff_alignment.py``.
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

SHORT = ["Bond", "Angle", "StrBnd", "Torsion", "OOP", "VDW", "Elec"]
LABELS = {"Elec": "Electro-\nstatics", "VDW": "VdW", "Angle": "Angle",
          "StrBnd": "Stretch\nbend", "Torsion": "Torsion", "OOP": "Out-of-\nplane",
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


def split_molecules() -> tuple[list[int], set[int]]:
    """Molecules whose MMFF94 energies align with r2SCAN-3c, and the rest."""
    a = pd.read_csv(OUT / "mmff94_alignment_per_molecule.tsv", sep="\t")
    fit = sorted(a[a.R2 >= R2_MIN].mol.tolist())
    return fit, set(a.mol) - set(fit)


def class_frames(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pair rows split by H-bond class: no H-bond, and anything else."""
    return {"noHB": d[d.cls == "noHB"], "other": d[d.cls != "noHB"]}


def fit_terms(g: pd.DataFrame, lam: float = 1e-2) -> np.ndarray:
    """Non-negative coefficients of the per-model logistic fit, one per term."""
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
        return (np.mean(np.logaddexp(0, z) - y * z) + lam * np.sum(c * c),
                np.r_[M.T @ r, X.T @ r + 2 * lam * c])

    return minimize(fg, np.r_[np.zeros(k), np.ones(7)], jac=True,
                    bounds=[(None, None)] * k + [(0, None)] * 7,
                    method="L-BFGS-B",
                    options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10}).x[-7:]


def amplitudes(q_all: pd.DataFrame, mols: set[int]) -> np.ndarray:
    """Mean absolute change of every term over the unique pairs of ``mols``."""
    unique = q_all.drop_duplicates(KEYS)
    return (unique[unique.molecule_number.isin(mols)]
            .groupby("molecule_number")[COLS]
            .apply(lambda g: g.abs().mean()).mean().values)



