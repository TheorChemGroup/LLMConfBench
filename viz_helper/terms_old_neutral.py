#!/usr/bin/env python3
"""Term counts old-vs-neutral for muse-glimmer:30b (one row per answer).

Reads predictions_muse-glimmer30b.jsonl from the paper bundles and the
geom-qm9-neutral bundles, counts chemical-term mentions in the explanation
(the "reasoning" field of the model's JSON answer, same source as
concreteness.py) and adds tau_f2.0 vs r2SCAN-3c. Writes a per-answer TSV plus
a per-molecule TSV aggregated over the 3 seeds; prints old-vs-neutral totals.

    python viz_helper/terms_old_neutral.py
    python viz_helper/terms_old_neutral.py -o /tmp/terms.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concreteness import HBOND_RE

from config.models_config import resolve_deltae_json
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import score_molecule

TAG = "muse-glimmer30b"
PROMPTS = {
    "old": ["geom-qm9/prompt_30x30_NEW", "geom-qm9/prompt_30x30_NEW_43",
            "geom-qm9/prompt_30x30_NEW_44"],
    "neutral": ["geom-qm9-neutral/prompt_30x30_NEW",
                "geom-qm9-neutral/prompt_30x30_NEW_43",
                "geom-qm9-neutral/prompt_30x30_NEW_44"],
}
SEEDS = {"prompt_30x30_NEW": 42, "prompt_30x30_NEW_43": 43, "prompt_30x30_NEW_44": 44}
LABELS_TSV = _REPO / "figures" / "seed_spread_heatmap_molecules.tsv"
OUT = _REPO / "figures" / "terms_old_neutral_museglimmer30b.tsv"

TERMS = {
    "n_hydrogen_bond": r"hydrogen bond",
    "n_hbond": r"h-bond",
    "n_ring_strain": r"ring strain",
    "n_steric": r"steric",
    "n_dipole": r"dipole",
    "n_eclips": r"eclips",
    "n_torsion": r"torsion",
    "n_angle": r"angle",
    "n_contact": r"contact",
    "n_distance": r"distance",
    "n_transannular": r"transannular",
    "n_pucker": r"pucker",
    "n_conjugat": r"conjugat",
    "n_aromatic": r"aromatic",
}
RES = {k: re.compile(v, re.IGNORECASE) for k, v in TERMS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    args = ap.parse_args()

    with LABELS_TSV.open(encoding="utf-8") as fh:
        labels = {r["molecule_id"]: r["label"]
                  for r in csv.DictReader(fh, delimiter="\t")}
    override = load_deltae_override(resolve_deltae_json("r2scan3c", None))

    rows: list[dict] = []
    for prompt, bundles in PROMPTS.items():
        for bundle in bundles:
            bdir = _REPO / bundle
            meta = augment_meta(
                load_metadata_index(bdir / "prompt_metadata.jsonl"),
                override, strict=True,
            )
            seed = SEEDS[Path(bundle).name]
            for line in (bdir / f"predictions_{TAG}.jsonl").open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                mol = str(r["molecule"])
                text = r.get("reason") or r.get("reasoning_raw") or ""
                tau = ""
                pred = r.get("prediction")
                if mol in meta and isinstance(pred, list):
                    s = score_molecule([str(x) for x in pred],
                                       meta[mol]["letter_to_deltaE_kcal_mol"], [2.0])
                    if s.get("valid"):
                        tau = f"{s['tau_f2.0']:.3f}"
                row = {"prompt": prompt, "seed": seed, "label": labels.get(mol, ""),
                       "molecule": mol, "tau_f2.0": tau, "words": len(text.split()),
                       "n_hbond_all": len(HBOND_RE.findall(text))}
                row.update({k: len(rx.findall(text)) for k, rx in RES.items()})
                rows.append(row)

    if not rows:
        print("No predictions found.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")

    count_cols = ["n_hbond_all", *TERMS]
    agg: dict[tuple, dict] = {}
    for r in rows:
        a = agg.setdefault((r["prompt"], r["label"], r["molecule"]),
                           {"n_seeds": 0, "taus": [], "words": 0,
                            **{k: 0 for k in count_cols}})
        a["n_seeds"] += 1
        if r["tau_f2.0"] != "":
            a["taus"].append(float(r["tau_f2.0"]))
        a["words"] += r["words"]
        for k in count_cols:
            a[k] += r[k]
    agg_rows = [{
        "prompt": p, "label": lab, "molecule": mol, "n_seeds": a["n_seeds"],
        "tau_f2.0": f"{sum(a['taus'])/len(a['taus']):.3f}" if a["taus"] else "",
        "words": a["words"], **{k: a[k] for k in count_cols},
    } for (p, lab, mol), a in sorted(
        agg.items(),
        key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 10**6, kv[0][2]))]
    agg_path = args.output.with_name(args.output.stem + "_agg" + args.output.suffix)
    with agg_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(agg_rows[0]), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote {agg_path} ({len(agg_rows)} rows)")

    for prompt in ("old", "neutral"):
        sub = [r for r in rows if r["prompt"] == prompt]
        taus = [float(r["tau_f2.0"]) for r in sub if r["tau_f2.0"] != ""]
        tot = {k: sum(r[k] for r in sub) for k in TERMS}
        keep = "  ".join(f"{k[2:]}={tot[k]}" for k in TERMS if tot[k])
        print(f"{prompt}: n={len(sub)} tau_f2.0={sum(taus)/len(taus):+.3f}  "
              f"n_hbond_all={sum(r['n_hbond_all'] for r in sub)}  {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
