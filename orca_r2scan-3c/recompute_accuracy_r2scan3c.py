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

from config.models_config import geom_seed_dirs, model_files_by_display

SEED_DIRS: list[tuple[str, Path]] = geom_seed_dirs()
MODEL_FILES: dict[str, str] = model_files_by_display("geom_benchmark")

PAIR_GAP_KCAL_MOL = 2.0
NORMAL_MAX_PAIRS = 10
HARD_MAX_PAIRS = 10
HARD_LARGE_PAIRS = 5
EXTREME_MAX_PAIRS = 10
MODES = ["normal", "hard", "extreme_hard"]


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
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(seed_dir / "prompt_metadata.jsonl"):
        out[str(row["molecule"])] = row
    return out


def load_answers(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        mol = row.get("molecule")
        rnk = row.get("ranking")
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


def all_pairs(energy: dict[int, float]) -> list[dict[str, Any]]:
    confs = sorted(energy.keys())
    pairs: list[dict[str, Any]] = []
    for i in range(len(confs)):
        for j in range(i + 1, len(confs)):
            a, b = confs[i], confs[j]
            ea, eb = energy[a], energy[b]
            if ea < eb:
                truth = "left"
            elif ea > eb:
                truth = "right"
            else:
                truth = "tie"
            pairs.append({
                "a_conf": a,
                "b_conf": b,
                "deltaE_pair_kcal_mol": round(abs(ea - eb), 3),
                "true_answer": truth,
            })
    return pairs


def select_mode_pairs(pairs_list: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    answerable = [p for p in pairs_list if p["deltaE_pair_kcal_mol"] > 0.0]
    large = [p for p in answerable if p["deltaE_pair_kcal_mol"] >= PAIR_GAP_KCAL_MOL]
    small = [p for p in answerable if p["deltaE_pair_kcal_mol"] < PAIR_GAP_KCAL_MOL]
    large.sort(key=lambda p: -p["deltaE_pair_kcal_mol"])
    small.sort(key=lambda p: p["deltaE_pair_kcal_mol"])
    if mode == "normal":
        return large[:NORMAL_MAX_PAIRS]
    if mode == "extreme_hard":
        return small[:EXTREME_MAX_PAIRS]
    if mode == "hard":
        chosen = large[:HARD_LARGE_PAIRS]
        missing = HARD_MAX_PAIRS - len(chosen)
        if missing > 0:
            chosen = chosen + small[:missing]
        return chosen
    raise ValueError(mode)


def model_pair_accuracy(
    conf_ranking: list[int], energy: dict[int, float], a: int, b: int
) -> float | None:
    if energy[a] == energy[b]:
        return None
    try:
        pa, pb = conf_ranking.index(a), conf_ranking.index(b)
    except ValueError:
        return None
    return 1.0 if (pa < pb) == (energy[a] < energy[b]) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltae-json", type=Path, default=Path("deltaE_r2scan3c_kcal_mol.json"))
    ap.add_argument("--seed-dirs", nargs=3, default=[s[1] for s in SEED_DIRS])
    ap.add_argument("--out", type=Path, default=Path("accuracy_r2scan3c.txt"))
    ap.add_argument("--energy-source", choices=["r2scan3c", "xtb"], default="r2scan3c",
                    help="r2scan3c: deltaE json from ORCA. xtb: letter_to_deltaE from metadata.")
    args = ap.parse_args()

    seed_dirs = [(s, Path(d)) for s, d in zip(["42", "43", "44"], args.seed_dirs)]

    if args.energy_source == "r2scan3c":
        deltaE_data: dict[str, dict[str, float]] = json.loads(args.deltae_json.read_text(encoding="utf-8"))
    else:
        deltaE_data = {}

    metas = {s: load_metas(d) for s, d in seed_dirs}
    answers = {
        s: {model: load_answers(d / fname) for model, fname in MODEL_FILES.items()}
        for s, d in seed_dirs
    }

    rankings: dict[str, dict[str, dict[str, list[int]]]] = {}
    for s, d in seed_dirs:
        for model in MODEL_FILES:
            for mol, labels in answers[s][model].items():
                row = metas[s].get(mol)
                if row is None:
                    continue
                cr = label_to_conf_ranking(row, labels)
                if cr is None:
                    continue
                rankings.setdefault(model, {}).setdefault(s, {})[mol] = cr

    accs: dict[str, dict[str, list[float]]] = {m: {mode: [] for mode in MODES} for m in MODEL_FILES}
    skipped: dict[str, dict[str, int]] = {}
    n_pairs_used: dict[str, int] = {mode: 0 for mode in MODES}

    for mol in (deltaE_data if args.energy_source == "r2scan3c" else metas["42"]):
        if args.energy_source == "r2scan3c":
            energy = {int(k): float(v) for k, v in deltaE_data[mol].items()}
        else:
            row = metas["42"].get(mol)
            if row is None:
                continue
            pub = row.get("public_label_to_conf_index", {})
            de = row.get("letter_to_deltaE_kcal_mol", {})
            energy = {}
            for lab, e in de.items():
                try:
                    energy[int(pub[lab])] = float(e)
                except (KeyError, ValueError):
                    continue
        if len(energy) < 2:
            continue
        pairs = all_pairs(energy)
        for mode in MODES:
            mode_pairs = select_mode_pairs(pairs, mode)
            n_pairs_used[mode] += len(mode_pairs)
            for model in MODEL_FILES:
                vals: list[float] = []
                for p in mode_pairs:
                    per_seed = []
                    for s in ("42", "43", "44"):
                        cr = rankings.get(model, {}).get(s, {}).get(mol)
                        if cr is None:
                            continue
                        acc = model_pair_accuracy(cr, energy, p["a_conf"], p["b_conf"])
                        if acc is not None:
                            per_seed.append(acc)
                    if per_seed:
                        vals.append(float(np.mean(per_seed)))
                if vals:
                    accs[model][mode].append(float(np.mean(vals)))
                else:
                    skipped.setdefault(model, {}).setdefault(mode, 0)
                    skipped[model][mode] += 1

    lines: list[str] = []
    lines.append("Pairwise accuracy vs r2SCAN-3c baseline (mean over molecules, 3 seeds):")
    lines.append(f"{'model':<24s} {'normal':>8s} {'hard':>8s} {'extreme_hard':>13s}")
    for model in MODEL_FILES:
        vals = []
        for mode in MODES:
            lst = accs[model][mode]
            vals.append(f"{np.mean(lst):.4f}" if lst else "N/A")
        lines.append(f"{model:<24s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>13s}")
    lines.append("")
    lines.append(f"Pairs per mode (total over molecules): "
                 f"{' '.join(f'{m}={n_pairs_used[m]}' for m in MODES)}")

    report = "\n".join(lines)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
