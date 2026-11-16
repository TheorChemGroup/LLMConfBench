"""Rebuild the per-run pair table (120 deg H-bond definition).

Reconstructs ``pair_runs_reliable.csv`` from the model answers, r2SCAN-3c
energies, conformer geometries and the MMFF94 term table. MMFF94 terms are read
from ``mmff94_terms_by_pair.csv`` (originally computed with OpenBabel); the
H-bond counts, answer positions and CRA are recomputed here.

    python viz_helper/build_pair_table.py [--out PATH]
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "pair_runs_reliable.csv"
sys.path[:0] = [str(REPO), str(REPO / "viz_helper")]

from config.models_config import (  # noqa: E402
    model_files_by_display,
    resolve_deltae_json,
)
from decoy_common import (  # noqa: E402
    FLOOR,
    bench_molecules,
    bonded_graph,
    hb_count_map,
    load_r2scan,
    read_xyz,
)
from metrics.kendall_tau_ranking import (  # noqa: E402
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import load_answers  # noqa: E402

BUNDLES = [("42", REPO / "geom-qm9/prompt_30x30_NEW"),
           ("43", REPO / "geom-qm9/prompt_30x30_NEW_43"),
           ("44", REPO / "geom-qm9/prompt_30x30_NEW_44")]
CONF = REPO / "geom-qm9/output_30_confs"
KEYS = ["molecule_number", "conf1", "conf2"]
TERMS = ["dE_Bond", "dE_Angle", "dE_StrBnd", "dE_Torsion", "dE_OOP",
         "dE_VDW", "dE_Electrostatic"]
BONDED = TERMS[:5]


def count_no_angle(atoms, bonded, h_max: float = 2.5) -> int:
    """H-bond count with the angle cutoff removed (distance only)."""
    xyz = {i: np.array(a[1:]) for i, a in enumerate(atoms)}
    n = 0
    for i, a in enumerate(atoms):
        if a[0] != "H":
            continue
        donors = [j for j in bonded.get(i, []) if atoms[j][0] in ("O", "N")]
        if len(donors) != 1:
            continue
        d = donors[0]
        for k, b in enumerate(atoms):
            if b[0] not in ("O", "N") or k in (i, d):
                continue
            r = float(np.linalg.norm(xyz[k] - xyz[i]))
            if 0 < r <= h_max:
                n += 1
    return n


def alt_counts(r2: dict) -> dict[tuple[str, int], int]:
    out = {}
    for mol in sorted(bench_molecules()):
        for ci in r2[mol]:
            atoms = read_xyz(CONF / mol / f"{mol}_conf_{ci}.xyz")
            out[(mol, int(ci))] = count_no_angle(atoms, bonded_graph(atoms))
    return out


def answer_positions() -> dict[tuple[str, str, str], dict[int, int]]:
    """(model, seed, molecule) -> {conf index: rank position}."""
    ov = load_deltae_override(resolve_deltae_json("r2scan3c", None))
    files = model_files_by_display("geom_benchmark")
    pos = {}
    for seed, bundle in BUNDLES:
        meta = augment_meta(load_metadata_index(bundle / "prompt_metadata.jsonl"),
                            ov, strict=True)
        label_to_conf = {
            mol: {int(k): int(v) for k, v in row["public_label_to_conf_index"].items()}
            for mol, row in meta.items()
        }
        for model, fname in files.items():
            path = bundle / fname
            if not path.is_file():
                continue
            for mol, ranking in load_answers(path).items():
                if mol not in meta:
                    continue
                pos[(model, seed, mol)] = {
                    label_to_conf[mol][int(lab)]: i for i, lab in enumerate(ranking)
                }
    return pos


def hb_abundance() -> dict[tuple[int, str, int], float]:
    """(molecule number, model, seed) -> relative H-bond mention abundance."""
    path = REPO / "figures/concreteness_vs_cra.tsv"
    if not path.is_file():
        print(f"warning: {path.name} missing; hb_abundance left empty")
        return {}
    t = pd.read_csv(path, sep="\t")
    x = t.assign(X=t.n_hbond / t.words)
    x["hb"] = x.X / x.model.map(x.groupby("model").X.mean())
    lut = pd.Series(x.hb.values, index=pd.MultiIndex.from_frame(
        x[["label", "model", "seed"]])).to_dict()
    return {k: round(v, 4) for k, v in lut.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    r2 = load_r2scan()
    counts = hb_count_map(r2, bench_molecules())
    alt = alt_counts(r2)
    terms = pd.read_csv(REPO / "mmff94_terms_by_pair.csv").set_index(KEYS)
    pos = answer_positions()
    ab = hb_abundance()
    labels = pd.read_csv(REPO / "figures/seed_spread_heatmap_molecules.tsv", sep="\t")
    num = dict(zip(labels.molecule_id, labels.label))
    models = model_files_by_display("geom_benchmark")

    rows = []
    for mol in sorted(bench_molecules(), key=lambda m: num[m]):
        de = {int(k): float(v) for k, v in r2[mol].items()}
        hb = {int(k): int(v) for k, v in counts[mol].items()}
        for c1, c2 in combinations(range(30), 2):
            dE = de[c2] - de[c1]
            if abs(dE) < FLOOR:
                continue
            key = (num[mol], c1, c2)
            if key not in terms.index:
                continue
            t = terms.loc[key]
            hbe, se = float(t.dE_Electrostatic), float(t[BONDED].sum())
            if hb[c1] == hb[c2]:
                cls = "equal_hb"
            else:
                cls = "favourable" if (hb[c1] - hb[c2]) * dE > 0 else "unfavourable"
            for model in models:
                for run_no, (seed, _) in enumerate(BUNDLES, start=1):
                    p = pos.get((model, seed, mol))
                    if p is None or c1 not in p or c2 not in p:
                        continue
                    lower = c1 if p[c1] < p[c2] else c2
                    correct = int(lower == (c1 if dE > 0 else c2))
                    rows.append({
                        "molecule_number": num[mol], "conf1": c1, "conf2": c2,
                        "hb1": hb[c1], "hb2": hb[c2],
                        "hb1_alt": alt[(mol, c1)], "hb2_alt": alt[(mol, c2)],
                        "hb_abundance": ab.get((num[mol], model, int(seed)),
                                               float("nan")),
                        "LLM": model, "run": run_no,
                        "dE_signed": round(dE, 4),
                        "HBE_signed": round(hbe, 6),
                        "SE_signed": round(se, 6),
                        "CRA": correct, "pair_class": cls,
                    })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows, {len(out.columns)} cols)")
    print("pair_class:", out.pair_class.value_counts().to_dict())
    print("hb_abundance NaN:", int(out.hb_abundance.isna().sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
