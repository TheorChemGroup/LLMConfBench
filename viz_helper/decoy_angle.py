"""H-bond geometry of the unconventional (decoy) pairs.

For every unconventional (decoy) pair, the strongest H-bond of the more
H-bonded conformer is located and its H...O/N distance and donor angle are
compared with the mean LLM accuracy on that pair.

    python viz_helper/decoy_angle.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(Path(__file__).parent)]

from config.models_config import geom_conformers_dir  # noqa: E402
from decoy_common import bonded_graph, hbond_contacts, read_xyz  # noqa: E402

KEYS = ["molecule_number", "conf1", "conf2"]
def geometry_table() -> pd.DataFrame:
    labels = pd.read_csv(OUT / "seed_spread_heatmap_molecules_fp.tsv", sep="\t")
    mol_id = dict(zip(labels.label, labels.molecule_id))
    conf = geom_conformers_dir()

    d = pd.read_csv(REPO / "pair_runs_reliable.csv")
    u = d.drop_duplicates(KEYS).copy()
    u["cls"] = np.where(
        u.hb1 == u.hb2, "eq",
        np.where(((u.hb1 > u.hb2) & (u.dE_signed > 0))
                 | ((u.hb2 > u.hb1) & (u.dE_signed < 0)), "fav", "dec"))
    u["rich"] = np.where(u.hb1 >= u.hb2, u.conf1, u.conf2)

    rows = []
    for r in u[u.cls == "dec"].itertuples():
        mid = mol_id[r.molecule_number]
        path = conf / mid / f"{mid}_conf_{r.rich}.xyz"
        if not path.is_file():
            continue
        atoms = read_xyz(path)
        cs = hbond_contacts(atoms, bonded_graph(atoms))
        if not cs:
            continue
        best = min(cs, key=lambda c: (c[0], -c[1]))
        rows.append({"molecule": r.molecule_number, "conf1": r.conf1,
                     "conf2": r.conf2, "rich": r.rich,
                     "dE": abs(r.dE_signed), "r": best[0], "angle": best[1]})
    g = pd.DataFrame(rows)
    acc = (d.groupby(KEYS).CRA.mean().rename("accuracy").reset_index()
           .rename(columns={"molecule_number": "molecule"}))
    return g.merge(acc, on=["molecule", "conf1", "conf2"])


def main() -> list[Path]:
    M = geometry_table()
    tsv = OUT / "decoy_hbond_geometry.tsv"
    M.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
    print(f"{len(M)} decoy pairs, {M.molecule.nunique()} molecules")
    return [tsv]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
