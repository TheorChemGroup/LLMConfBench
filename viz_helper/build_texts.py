"""Rebuild the ``texts/`` explanation dumps from the committed prompt bundles.

    python viz_helper/build_texts.py

``texts/`` holds one JSONL per model and seed with the model's own reasoning
text plus the run's reliable-pair Kendall's tau.  Those texts are fully derived
from the committed ``geom-qm9/prompt_30x30_NEW*/predictions_<tag>.jsonl`` and
``prompt_metadata.jsonl``, so this script recreates them and the 65 MB dump
does not need to be shipped.

Writes:
  texts/seed{42,43,44}/<model>.jsonl   molecule, smiles, chain_of_thought, reasoning, tau
  texts/seed_avg/<model>.jsonl         + taus/n_seeds dicts (tau = mean of valid runs)

tau is the reliable-pair Kendall's tau at TAU_FLOOR (2.0 kcal/mol), rounded to
4 decimals, ``None`` when the model returned no valid ranking for that molecule.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path[:0] = [str(_REPO), str(_HERE)]

from config.models_config import (
    TAU_FLOOR,
    geom_seed_paths,
    models_in_group,
    mols_md_path,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import score_molecule

SEED_NAMES = ("42", "43", "44")
TAU_KEY = f"tau_f{TAU_FLOOR}"
MOL_TABLE = _REPO / "figures" / "r2scan3c" / "seed_spread_heatmap_molecules.tsv"


_FF_SMILES_OVERRIDE = {
    "C_C_H__C_O_CO_CH__NH_": "C[C@H](C=O)COC=N",
    "C_C___NH__OCCCCO": "CC(=N)OCCCCO",
}


def smiles_by_molecule() -> dict[str, str]:
    """Sanitized-SMILES folder id -> SMILES, from MDs/MOLS.md."""
    out: dict[str, str] = {}
    for smi in ast.literal_eval(mols_md_path().read_text(encoding="utf-8").strip()):
        key = "".join(c if c.isalnum() else "_" for c in smi)
        out[key] = _FF_SMILES_OVERRIDE.get(key, smi)
    return out


def molecule_order() -> list[str]:
    """Cohort order: the heatmap table when present, else sorted molecule ids."""
    if MOL_TABLE.is_file():
        rows = [
            line.split("\t")[1]
            for line in MOL_TABLE.read_text(encoding="utf-8").splitlines()[1:]
            if line.strip()
        ]
        if rows:
            return rows
    return sorted(smiles_by_molecule())


def run_tau(ranking, letter_to_de) -> float | None:
    if not ranking or not letter_to_de:
        return None
    scored = score_molecule([str(x) for x in ranking], letter_to_de, [TAU_FLOOR])
    if not scored.get("valid"):
        return None
    value = scored.get(TAU_KEY)
    return round(float(value), 4) if value is not None else None


def tag_of(model) -> str:
    name = model.answers
    return name[len("answers_") : -len(".jsonl")]


def build(out_dir: Path) -> list[Path]:
    models = models_in_group("geom_benchmark")
    smiles = smiles_by_molecule()
    molecules = molecule_order()
    override = load_deltae_override(resolve_deltae_json("r2scan3c", None))

    written: list[Path] = []
    per_model: dict[str, dict[str, dict]] = {}

    for seed, bundle in zip(SEED_NAMES, geom_seed_paths()):
        meta = augment_meta(
            load_metadata_index(bundle / "prompt_metadata.jsonl"),
            override,
            strict=True,
        )
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            pred_path = bundle / f"predictions_{tag_of(model)}.jsonl"
            if not pred_path.is_file():
                print(f"  [skip] {model.id}: no {pred_path.name}")
                continue
            preds = {
                json.loads(line)["molecule"]: json.loads(line)
                for line in pred_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            exposes_cot = any(
                p.get("reasoning_content") is not None for p in preds.values()
            )
            records = []
            for mol in molecules:
                pred = preds.get(mol, {})
                letter_to_de = meta.get(mol, {}).get("letter_to_deltaE_kcal_mol")
                rec = {"molecule": mol, "smiles": smiles.get(mol, "")}
                if exposes_cot:
                    rec["chain_of_thought"] = pred.get("reasoning_content")
                if pred.get("reason") is not None:
                    rec["reasoning"] = pred["reason"]
                rec["tau"] = run_tau(pred.get("prediction"), letter_to_de)
                records.append(rec)
                per_model.setdefault(model.id, {})[mol] = per_model.setdefault(
                    model.id, {}
                ).get(mol, {})
                per_model[model.id][mol][seed] = rec
            path = seed_dir / f"{model.id}.jsonl"
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8",
            )
            written.append(path)
        print(f"seed {seed}: wrote {len(models)} model files -> {seed_dir}")

    avg_dir = out_dir / "seed_avg"
    avg_dir.mkdir(parents=True, exist_ok=True)
    for model_id, by_mol in per_model.items():
        records = []
        for mol in molecules:
            runs = by_mol.get(mol, {})
            taus = {s: runs.get(s, {}).get("tau") for s in SEED_NAMES}
            valid = [v for v in taus.values() if v is not None]
            rec = {
                "molecule": mol,
                "smiles": smiles.get(mol, ""),
                "tau": round(sum(valid) / len(valid), 4) if valid else None,
                "n_seeds": len(valid),
                "taus": taus,
            }
            cot = {s: runs.get(s, {}).get("chain_of_thought") for s in SEED_NAMES}
            if any("chain_of_thought" in runs.get(s, {}) for s in SEED_NAMES):
                rec["chain_of_thought"] = cot
            rea = {s: runs.get(s, {}).get("reasoning") for s in SEED_NAMES}
            if any(v is not None for v in rea.values()):
                rec["reasoning"] = rea
            records.append(rec)
        path = avg_dir / f"{model_id}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
        written.append(path)
    print(f"seed_avg: wrote {len(per_model)} model files -> {avg_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "texts")
    args = ap.parse_args()
    written = build(args.out_dir)
    print(f"Done: {len(written)} files under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
