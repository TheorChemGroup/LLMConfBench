"""MMFF94 energy-term decomposition per reliable conformer pair.

Terms come from Open Babel's MMFF94 on the RDKit template (SMILES bond orders), so
the force field sees the same chemistry as the RDKit baseline; totals agree with
RDKit to <0.01 kcal/mol. Differences are stored as dE_t = E_t(conf2) - E_t(conf1);
readers orient them along the r2SCAN-3c ordering through ``dE_signed``.

Writes ``mmff94_terms_by_pair.csv``.

    python viz_helper/mmff94_terms.py [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from rdkit import Chem  # noqa: F401  (must precede openbabel: libstdc++ clash)

from openbabel import openbabel as ob

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "mmff94_terms_by_pair.csv"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(REPO / "viz_helper")]

from decoy_common import FLOOR, bench_molecules, load_r2scan  # noqa: E402
from plot_metrics_vs_release import (  # noqa: E402
    MMFF94_NAME,
    _build_template_with_conformers,
)

TERMS = ("E_Bond", "E_Angle", "E_StrBnd", "E_Torsion", "E_OOP",
         "E_VDW", "E_Electrostatic")
CONF = REPO / "geom-qm9/output_30_confs"


def conformer_terms(folder: str) -> dict[int, dict[str, float]]:
    """MMFF94 term energies of every conformer of one molecule (Open Babel)."""
    built = _build_template_with_conformers(CONF / folder, MMFF94_NAME)
    if built is None:
        raise SystemExit(f"{folder}: RDKit template build failed")
    template, conf_of = built
    conv = ob.OBConversion()
    conv.SetInFormat("mol")
    ff = ob.OBForceField.FindForceField("MMFF94")
    out = {}
    for cid in range(template.GetNumConformers()):
        obmol = ob.OBMol()
        if not conv.ReadString(obmol, Chem.MolToMolBlock(template, confId=cid)):
            raise SystemExit(f"{folder} conf {cid}: molblock read failed")
        if not ff.Setup(obmol):
            raise SystemExit(f"{folder} conf {cid}: MMFF94 setup failed")
        out[int(conf_of[cid])] = {t: getattr(ff, t)() for t in TERMS}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    labels_path = REPO / "figures/seed_spread_heatmap_molecules_fp.tsv"
    if not labels_path.is_file():
        raise SystemExit(f"{labels_path} is missing: run "
                         "viz_helper/seed_spread_heatmap.py first")
    labels = pd.read_csv(labels_path, sep="\t")
    num_of = dict(zip(labels.molecule_id, labels.label))
    energy = load_r2scan()

    rows = []
    for folder in bench_molecules():
        terms = conformer_terms(folder)
        de = {int(k): float(v) for k, v in energy[folder].items()}
        for c1, c2 in combinations(range(30), 2):
            if abs(de[c2] - de[c1]) < FLOOR:
                continue
            rows.append({"molecule_number": num_of[folder], "conf1": c1, "conf2": c2,
                         **{f"dE_{t[2:]}": terms[c2][t] - terms[c1][t] for t in TERMS}})

    out = pd.DataFrame(rows).sort_values(["molecule_number", "conf1", "conf2"])
    out = out.round(6)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
