#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import geom_seed_paths, model_files_by_display

FLOORS = [0.5, 1.0, 1.5, 2.0]


def load_deltae_json(path: Path) -> dict[str, dict[str, float]]:
    return {mol: {int(k): float(v) for k, v in de.items()}
            for mol, de in json.loads(path.read_text(encoding="utf-8")).items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_metas(seed_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(r["molecule"]): r for r in load_jsonl(seed_dir / "prompt_metadata.jsonl")}


def load_answers(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in load_jsonl(path):
        mol = r.get("molecule")
        rnk = r.get("ranking")
        if mol is not None and rnk is not None:
            out[str(mol)] = [str(x) for x in rnk]
    return out


def label_to_conf_ranking(row: dict[str, Any], labels: list[str]) -> list[int] | None:
    pub_map = row.get("public_label_to_conf_index", {})
    n = row.get("n_conformers", 0)
    if len(labels) != n:
        return None
    try:
        conf_ranking = [int(pub_map[lab]) for lab in labels]
    except (KeyError, ValueError):
        return None
    if len(set(conf_ranking)) != n:
        return None
    return conf_ranking


def reliable_tau(energy: dict[int, float], conf_ranking: list[int], floor: float) -> float | None:
    """Kendall tau restricted to pairs with |dE| >= floor."""
    confs = sorted(energy.keys())
    pred_rank = {c: i for i, c in enumerate(conf_ranking)}
    agree = disc = 0
    for a in range(len(confs)):
        for b in range(a + 1, len(confs)):
            ca, cb = confs[a], confs[b]
            if abs(energy[ca] - energy[cb]) < floor:
                continue
            if ca not in pred_rank or cb not in pred_rank:
                continue
            truth = (energy[ca] < energy[cb]) - (energy[ca] > energy[cb])
            pred = (pred_rank[ca] < pred_rank[cb]) - (pred_rank[ca] > pred_rank[cb])
            if truth == 0:
                continue
            if pred == truth:
                agree += 1
            else:
                disc += 1
    n = agree + disc
    return float((agree - disc) / n) if n > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r2scan-json", type=Path,
                    default=_REPO / "orca_r2scan-3c" / "deltaE_r2scan3c_kcal_mol.json")
    ap.add_argument("--ccsdt-json", type=Path,
                    default=_REPO / "orca_ccsdt" / "deltaE_dlpno_ccsdt_kcal_mol.json")
    ap.add_argument("--seed-dirs", nargs=3, default=None,
                    help="3 seed bundles (default from models.yaml)")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "orca_ccsdt" / "compare_baselines_subset.txt")
    args = ap.parse_args()

    seed_dir_args = args.seed_dirs if args.seed_dirs is not None else [str(p) for p in geom_seed_paths()]
    seed_dirs = [(s, Path(d)) for s, d in zip(["42", "43", "44"], seed_dir_args)]
    model_files = model_files_by_display("geom_benchmark")
    r2 = load_deltae_json(args.r2scan_json)
    cc = load_deltae_json(args.ccsdt_json)
    mols = sorted(set(r2) & set(cc))
    if not mols:
        print("No molecules shared between the two baselines.", file=sys.stderr)
        return 1
    print(f"{len(mols)} subset molecules shared between baselines")

    metas = {s: load_metas(d) for s, d in seed_dirs}
    answers = {s: {m: load_answers(d / f) for m, f in model_files.items()}
               for s, d in seed_dirs}

    lines: list[str] = []
    lines.append(f"Baseline agreement r2SCAN-3c vs DLPNO-CCSD(T) on {len(mols)} molecules:")
    lines.append(f"{'floor kcal/mol':>14s} {'pairs':>7s} {'agree':>8s} {'disagree':>9s} {'agree%':>8s}")
    for floor in FLOORS:
        tot_a = tot_d = 0
        for mol in mols:
            confs = sorted(r2[mol].keys())
            for a in range(len(confs)):
                for b in range(a + 1, len(confs)):
                    ca, cb = confs[a], confs[b]
                    if abs(r2[mol][ca] - r2[mol][cb]) < floor:
                        continue
                    dr = (r2[mol][ca] < r2[mol][cb]) - (r2[mol][ca] > r2[mol][cb])
                    dc = (cc[mol][ca] < cc[mol][cb]) - (cc[mol][ca] > cc[mol][cb])
                    if dr == 0:
                        continue
                    if dc == dr:
                        tot_a += 1
                    else:
                        tot_d += 1
        pct = 100.0 * tot_a / (tot_a + tot_d) if (tot_a + tot_d) else float("nan")
        lines.append(f"{floor:>14.1f} {tot_a + tot_d:>7d} {tot_a:>8d} {tot_d:>9d} {pct:>7.1f}%")

    lines.append("")
    lines.append(
        "Model-ranking agreement (r2SCAN-3c vs CCSD(T) scores; "
        "LLM-cohort mols in subset × 3 seeds):"
    )
    lines.append(f"{'floor':>8s} {'Spearman rho':>13s} {'Kendall tau':>12s}")
    from scipy.stats import kendalltau as kt
    from scipy.stats import spearmanr
    for floor in FLOORS:
        r2_scores: dict[str, float] = {}
        cc_scores: dict[str, float] = {}
        for model in model_files:
            r2_taus: list[float] = []
            cc_taus: list[float] = []
            for s, d in seed_dirs:
                meta = metas[s]
                ans = answers[s][model]
                for mol in mols:
                    row = meta.get(mol)
                    if row is None:
                        continue
                    cr = label_to_conf_ranking(row, ans.get(mol, []))
                    if cr is None:
                        continue
                    if mol in r2:
                        t = reliable_tau(r2[mol], cr, floor)
                        if t is not None:
                            r2_taus.append(t)
                    if mol in cc:
                        t = reliable_tau(cc[mol], cr, floor)
                        if t is not None:
                            cc_taus.append(t)
            r2_scores[model] = float(np.mean(r2_taus)) if r2_taus else float("nan")
            cc_scores[model] = float(np.mean(cc_taus)) if cc_taus else float("nan")
        vals_r = np.array([r2_scores[m] for m in model_files], dtype=float)
        vals_c = np.array([cc_scores[m] for m in model_files], dtype=float)
        rho, _ = spearmanr(vals_r, vals_c)
        tau, _ = kt(vals_r, vals_c)
        lines.append(f"{floor:>8.1f} {rho:>13.4f} {tau:>12.4f}")

    report = "\n".join(lines)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
