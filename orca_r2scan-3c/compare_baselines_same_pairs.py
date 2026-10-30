from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED_DIRS: list[tuple[str, Path]] = [
    ("42", Path("prompt_30x30_NEW")),
    ("43", Path("prompt_30x30_NEW_43")),
    ("44", Path("prompt_30x30_NEW_44")),
]

MODEL_FILES: dict[str, str] = {
    "Claude Opus 5": "answers_claudeopus5.jsonl",
    "Claude Sonnet 5": "answers_claudesonnet5.jsonl",
    "GPT 5.6 Sol": "answers_gpt56sol.jsonl",
    "Gemini 3.6 Flash": "answers_gemini36flash.jsonl",
    "Kimi K3": "answers_kimik3.jsonl",
    "Qwen 3.7 Max": "answers_qwen37max.jsonl",
    "Qwen 3.6 27B": "answers_qwen3627b.jsonl",
    "Qwen 3.5 Plus": "answers_qwen35plus.jsonl",
    "DeepSeek V4 Flash 0731": "answers_deepseekv4flash.jsonl",
    "DeepSeek R1 32B": "answers_deepseekr132b.jsonl",
}

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
    return {str(r["molecule"]): r for r in load_jsonl(seed_dir / "prompt_metadata.jsonl")}


def load_answers(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        mol, rnk = row.get("molecule"), row.get("ranking")
        if mol is not None and rnk is not None:
            out[str(mol)] = [str(x) for x in rnk]
    return out


def label_to_conf_ranking(row: dict[str, Any], labels: list[str]) -> list[int] | None:
    pub_map = row.get("public_label_to_conf_index", {})
    n = row.get("n_conformers", 0)
    if len(labels) != n:
        return None
    try:
        cr = [int(pub_map[lab]) for lab in labels]
    except (KeyError, ValueError):
        return None
    if len(set(cr)) != n:
        return None
    return cr


def xtb_energy(row: dict[str, Any]) -> dict[int, float]:
    pub = row.get("public_label_to_conf_index", {})
    out: dict[int, float] = {}
    for lab, e in row.get("letter_to_deltaE_kcal_mol", {}).items():
        try:
            out[int(pub[lab])] = float(e)
        except (KeyError, ValueError):
            continue
    return out


def all_pairs(energy: dict[int, float]) -> list[dict[str, Any]]:
    confs = sorted(energy.keys())
    pairs: list[dict[str, Any]] = []
    for i in range(len(confs)):
        for j in range(i + 1, len(confs)):
            a, b = confs[i], confs[j]
            ea, eb = energy[a], energy[b]
            truth = "left" if ea < eb else ("right" if ea > eb else "tie")
            pairs.append({
                "a_conf": a, "b_conf": b,
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


def pair_accuracy(cr: list[int], energy: dict[int, float], a: int, b: int) -> float | None:
    if energy.get(a) is None or energy.get(b) is None:
        return None
    if energy[a] == energy[b]:
        return None
    try:
        pa, pb = cr.index(a), cr.index(b)
    except ValueError:
        return None
    return 1.0 if (pa < pb) == (energy[a] < energy[b]) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltae-json", type=Path, default=Path("deltaE_r2scan3c_kcal_mol.json"))
    ap.add_argument("--seed-dirs", nargs=3, default=[s[1] for s in SEED_DIRS])
    ap.add_argument("--out", type=Path, default=Path("compare_same_pairs.txt"))
    args = ap.parse_args()

    seed_dirs = [(s, Path(d)) for s, d in zip(["42", "43", "44"], args.seed_dirs)]
    r2 = json.loads(args.deltae_json.read_text(encoding="utf-8"))
    metas = {s: load_metas(d) for s, d in seed_dirs}
    answers = {
        s: {m: load_answers(d / f) for m, f in MODEL_FILES.items()}
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

    stats: dict[str, dict[str, dict[str, list[float]]]] = {}
    n_pairs: dict[str, int] = {m: 0 for m in MODES}
    for mol, de_map in r2.items():
        e_r2 = {int(k): float(v) for k, v in de_map.items()}
        if len(e_r2) < 2:
            continue
        row42 = metas["42"].get(mol)
        if row42 is None:
            continue
        e_xtb = xtb_energy(row42)

        for mode in MODES:
            mode_pairs = select_mode_pairs(all_pairs(e_r2), mode)
            mode_pairs = [
                p for p in mode_pairs
                if e_r2.get(p["a_conf"]) is not None
                and e_r2.get(p["b_conf"]) is not None
                and e_xtb.get(p["a_conf"]) is not None
                and e_xtb.get(p["b_conf"]) is not None
                and e_r2[p["a_conf"]] != e_r2[p["b_conf"]]
                and e_xtb[p["a_conf"]] != e_xtb[p["b_conf"]]
            ]
            n_pairs[mode] += len(mode_pairs)
            for model in MODEL_FILES:
                per_seed_r2: list[float] = []
                per_seed_xtb: list[float] = []
                for p in mode_pairs:
                    a, b = p["a_conf"], p["b_conf"]
                    for s in ("42", "43", "44"):
                        cr = rankings.get(model, {}).get(s, {}).get(mol)
                        if cr is None:
                            continue
                        ar2 = pair_accuracy(cr, e_r2, a, b)
                        if ar2 is not None:
                            per_seed_r2.append(ar2)
                        axtb = pair_accuracy(cr, e_xtb, a, b)
                        if axtb is not None:
                            per_seed_xtb.append(axtb)
                stats.setdefault(model, {}).setdefault(mode, {})
                if per_seed_r2:
                    stats[model][mode]["r2scan3c"] = stats[model][mode].get("r2scan3c", []) + [float(np.mean(per_seed_r2))]
                if per_seed_xtb:
                    stats[model][mode]["xtb"] = stats[model][mode].get("xtb", []) + [float(np.mean(per_seed_xtb))]

    lines: list[str] = []
    lines.append("SAME pairs (selected by r2SCAN-3c), accuracy with both baselines:")
    lines.append(f"{'model':<24s} {'normal':^22s} {'hard':^22s} {'extreme_hard':^22s}")
    lines.append(f"{'':24s} {'r2scan3c':>9s} {'xtb':>9s} {'r2scan3c':>9s} {'xtb':>9s} {'r2scan3c':>9s} {'xtb':>9s}")
    for model in MODEL_FILES:
        cells = []
        for mode in MODES:
            d = stats.get(model, {}).get(mode, {})
            cells.append(f"{np.mean(d['r2scan3c']):9.4f}" if d.get("r2scan3c") else "      N/A")
            cells.append(f"{np.mean(d['xtb']):9.4f}" if d.get("xtb") else "      N/A")
        lines.append(f"{model:<24s} {' '.join(cells)}")
    lines.append("")
    lines.append(f"Pairs per mode: {' '.join(f'{m}={n_pairs[m]}' for m in MODES)}")
    lines.append("Note: both columns use the identical pair set (xTB-tie and r2-tie pairs dropped).")

    report = "\n".join(lines)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
