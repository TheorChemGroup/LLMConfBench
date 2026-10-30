"""Chemical-concept vocabulary for explanation texts.

Counts non-H-bond chemical concepts per explanation and writes
``figures/concept_counts_vs_cra.tsv``. Rows, CRA values, word counts and the
H-bond count are taken from the concreteness table, so the H-bond axis here is
exactly the one in ``FullPlot.png``.

    python viz_helper/concepts.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "viz_helper"))

from concreteness import HBOND_RE, _reasoning_text, load_setup

TEXTS_DIR = Path("texts")
LABELS_TSV = Path("figures/seed_spread_heatmap_molecules.tsv")
CONCRETENESS_TSV = Path("figures/concreteness_vs_cra.tsv")
TABLE_TSV = Path("figures/concept_counts_vs_cra.tsv")
SEEDS = ("42", "43", "44")

# H-bonding is the concreteness definition; other concepts are independent.
CONCEPTS = {
    "H-bonding": HBOND_RE.pattern,
    "Steric clash / crowding": r"steric|clash|crowd|congest|repuls|pauli|hindrance",
    "Ring strain": (
        r"ring strain|angle strain|baeyer|small ring|three-membered|four-membered|"
        r"cycloprop|cyclobut|ring tension|strain energy"
    ),
    "Torsion / eclipsing": (
        r"torsion|torsional|eclips|stagger|\bgauche\b|\bsyn\b|\banti\b(?!-?bond)|"
        r"dihedral|periplanar|pucker|ring[ -]flip|half-chair|twist-boat|\bchair\b|\bboat\b"
    ),
    "Dipole / electrostatics": (
        r"dipole|electrostatic|charge separation|\bpolar\b|polarity|polariz|"
        r"non-?polar|partial charge|charge distribution"
    ),
    "Dispersion / vdW": (
        r"dispersion|london|van[ -]der[ -]waals|\bvdw\b|π[ -]?stack|pi[ -]?stack"
    ),
    "Hyperconjugation / anomeric": (
        r"hyperconjugat|anomeric|stereoelectronic|sigma\s*\*|σ\s*\*|n\s*→\s*σ"
    ),
    "Conjugation / aromaticity": r"conjugat|aromatic|resonance|delocali|π|pi[ -]system",
    "Nonbonded / transannular": r"nonbonded|non-bonded|through[ -]space|transannular",
    "1,3-Diaxial": r"1[,.]3[ -]?diaxial|diaxial",
    "Flexibility / entropy": (
        r"entropy|flexib|conformational freedom|degrees of freedom|rotatable|\brigid"
    ),
    "Quantitative geometry": r"\d+(?:\.\d+)?\s*(?:å|angstrom|degrees?|°)",
    "Solvation / environment": (
        r"solvat|solvent|aqueous|water|dielectric|implicit solvent|hydrat"
    ),
    "Symmetry": r"symmetr",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in CONCEPTS.items()}
COMPILED["H-bonding"] = HBOND_RE


def label_to_molecule(labels_tsv: Path = LABELS_TSV) -> dict[int, str]:
    out = {}
    for line in labels_tsv.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            out[int(parts[0])] = parts[1]
    return out


def _text_index(model_id: str, seed: str, texts_dir: Path) -> dict[str, str]:
    path = texts_dir / f"seed{seed}" / f"{model_id}.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["molecule"]] = _reasoning_text(rec)
    return out


def build_table(
    *,
    texts_dir: Path = TEXTS_DIR,
    labels_tsv: Path = LABELS_TSV,
    concreteness_tsv: Path = CONCRETENESS_TSV,
    out_tsv: Path = TABLE_TSV,
) -> pd.DataFrame:
    base = pd.read_csv(concreteness_tsv, sep="\t")
    label_map = label_to_molecule(labels_tsv)
    caches: dict[tuple[str, str], dict[str, str]] = {}

    rows = []
    for r in base.itertuples(index=False):
        molecule = label_map.get(int(r.label))
        if molecule is None:
            continue
        key = (r.model_id, str(r.seed))
        if key not in caches:
            caches[key] = _text_index(*key, texts_dir)
        text = caches[key].get(molecule, "")

        row = {
            "model_id": r.model_id,
            "model": r.model,
            "seed": str(r.seed),
            "molecule": molecule,
            "label": int(r.label),
            "tau": r.tau,
            "words": r.words,
        }
        for concept, rx in COMPILED.items():
            row[concept] = len(rx.findall(text))
        # Must equal the concreteness H-bond count row by row.
        assert row["H-bonding"] == int(r.n_hbond), (key, molecule)
        row["n_H-bonding"] = row["H-bonding"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(
        f"{len(df)} explanations, {df.model.nunique()} models, "
        f"{df.molecule.nunique()} molecules -> {out_tsv}"
    )
    return df


def main() -> None:
    models, _mol_to_label, _mol_labels = load_setup()
    build_table()
    print(f"{len(models)} models available; concepts: {', '.join(CONCEPTS)}")


if __name__ == "__main__":
    main()
