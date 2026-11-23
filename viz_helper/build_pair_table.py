"""Rebuild the per-run pair table (120 deg H-bond definition).

Reconstructs ``pair_runs_reliable.csv`` from the model answers, r2SCAN-3c
energies, conformer geometries and the MMFF94 term table. MMFF94 terms are read
from ``mmff94_terms_by_pair.csv`` (originally computed with OpenBabel); the
H-bond counts, answer positions and CRA are recomputed here.

    python viz_helper/build_pair_table.py [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

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
    hb_count_map,
    load_r2scan,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    r2 = load_r2scan()
    counts = hb_count_map(r2, bench_molecules())
    terms = pd.read_csv(REPO / "mmff94_terms_by_pair.csv").set_index(KEYS)
    pos = answer_positions()
    labels = pd.read_csv(REPO / "figures/seed_spread_heatmap_molecules_fp.tsv", sep="\t")
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
                        "LLM": model, "run": run_no,
                        "dE_signed": round(dE, 4), "CRA": correct,
                    })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows, {len(out.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
