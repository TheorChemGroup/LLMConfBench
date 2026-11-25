"""Reference line for the CRA-vs-year panels: the H-bond counting rule.

The rule ranks the conformer with more intramolecular H-bonds lower; ties count
as 0.5. Values use the same macro CRA as the force-field lines: per-molecule CRA
on the panel's pair table, averaged over molecules.

    python viz_helper/hb_rule_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "viz_helper")]

KEYS = ["molecule_number", "conf1", "conf2"]

REFS = (("H-bond count rule", "#6d4c41", (0, (1, 1.5))),)
RULE = "H-bond count rule"


def all_pairs() -> dict[str, list[tuple[int, int]]]:
    """Truth-ordered pairs at |dE| >= 2 kcal/mol, keyed by molecule folder."""
    labels = pd.read_csv(REPO / "figures/seed_spread_heatmap_molecules_fp.tsv", sep="\t")
    folder = dict(zip(labels.label.astype(int), labels.molecule_id))
    d = pd.read_csv(REPO / "pair_runs_reliable.csv").drop_duplicates(KEYS)
    out: dict[str, list[tuple[int, int]]] = {}
    for mol, g in d.groupby("molecule_number"):
        out[folder[int(mol)]] = [
            (int(r.conf1), int(r.conf2)) if r.dE_signed > 0
            else (int(r.conf2), int(r.conf1))
            for r in g.itertuples()
        ]
    return out


def per_molecule(table: dict, counts: dict) -> list[float]:
    """CRA of the H-bond counting rule, one value per molecule.

    ``table``: {folder: [(lo, hi)]}, lo = lower-energy conformer.
    ``counts``: {folder: {conf: n_H-bonds}} from ``decoy_common.hb_count_map``.
    Pairs with equal counts count as 0.5 (random choice).
    """
    out = []
    for mol, pairs in table.items():
        cnt = {int(k): v for k, v in counts.get(mol, {}).items()}
        rows = [1.0 if cnt[lo] > cnt[hi] else 0.0 if cnt[lo] < cnt[hi] else 0.5
                for lo, hi in pairs if lo in cnt and hi in cnt]
        if rows:
            out.append(2 * float(np.mean(rows)) - 1)
    return out


def values(table: dict, counts: dict) -> list[tuple[str, float]]:
    """[(name, macro CRA)] for the H-bond counting rule on a pair table."""
    per_mol = per_molecule(table, counts)
    return [(RULE, float(np.mean(per_mol)))] if per_mol else []


def draw(ax, table: dict, counts: dict, *, lw: float = 1.1) -> None:
    by_name = dict(values(table, counts))
    for name, color, ls in REFS:
        if name in by_name:
            ax.axhline(by_name[name], color=color, ls=ls, lw=lw, zorder=1)


def ref_handles() -> list:
    """Legend entries for the reference lines (force fields + H-bond rule)."""
    from matplotlib.lines import Line2D

    from viz.plot_style import (REF_GFNFF, REF_MMFF, REF_MMFF_LINESTYLE,
                                REF_RANDOM, REF_UFF)

    handles = [
        Line2D([], [], color=REF_RANDOM, ls="--", lw=1.1, label="Random"),
        Line2D([], [], color=REF_UFF, ls=":", lw=1.3, label="UFF"),
        Line2D([], [], color=REF_GFNFF, ls="-.", lw=1.3, label="GFN-FF"),
        Line2D([], [], color=REF_MMFF, ls=REF_MMFF_LINESTYLE, lw=1.3,
               label="MMFF94"),
    ]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.1, label=n)
                for n, c, ls in REFS]
    return handles


def main() -> int:
    from decoy_common import (bench_molecules, equal_hbond_pair_tables,
                              hb_count_map, hbond_pair_tables, load_r2scan)

    r2scan = load_r2scan()
    counts = hb_count_map(r2scan, bench_molecules())
    hb = hbond_pair_tables(r2scan, counts)
    tables = {
        "all": all_pairs(),
        "fair": equal_hbond_pair_tables(r2scan, counts),
        "fav": hb["fav"],
        "dec": hb["dec"],
        "noHB": hb["noHB"],
    }
    for kind, table in tables.items():
        print(f"{kind:5s} " + "  ".join(f"{n} {v:+.3f}" for n, v in values(table, counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
