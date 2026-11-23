from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import model_files_by_display  # noqa: E402

# Every model in the benchmark, not the xTB subset.
MODEL_FILES = model_files_by_display("geom_benchmark")

from orca_pairs import (  # noqa: E402
    MODES,
    SEED_DIRS,
    all_pairs,
    label_to_conf_ranking,
    load_answers,
    load_metas,
    select_mode_pairs,
)


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
    for s, _d in seed_dirs:
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
