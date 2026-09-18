#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from config.models_config import add_baseline_arguments, resolve_deltae_json
from metrics.kendall_tau_ranking import (
    augment_meta,
    label_sort_key,
    load_deltae_override,
    load_metadata_index,
)

DEFAULT_FLOORS = [0.5, 1.0, 1.5, 2.0]


def load_answers(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        mol = str(obj["molecule"])
        rnk = obj.get("ranking")
        if mol and isinstance(rnk, list):
            out[mol] = [str(x) for x in rnk]
    return out


def load_valid_answers(path: Path, meta: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Load rankings that are exact permutations of each molecule's labels."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i, out = 0, {}
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            nxt = text.find('{"molecule"', i + 1)
            i = nxt if nxt != -1 else len(text)
            continue
        i = end
        mol = obj.get("molecule")
        rnk = obj.get("ranking")
        if mol is None or rnk is None or mol not in meta:
            continue
        rnk = [str(x) for x in rnk]
        keys = set(meta[mol]["letter_to_deltaE_kcal_mol"].keys())
        if set(rnk) == keys and len(rnk) == len(keys):
            out[str(mol)] = rnk
    return out


def reliable_pairs(de: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(de)
    iu = np.triu_indices(n, 1)
    keep = np.abs(de[iu[0]] - de[iu[1]]) >= floor
    return iu[0][keep], iu[1][keep]


def reliable_tau(de: np.ndarray, pred_rank: np.ndarray, floor: float) -> tuple[float, int]:
    i, j = reliable_pairs(de, floor)
    n_pairs = len(i)
    if n_pairs == 0:
        return float("nan"), 0
    agree = np.sum(np.sign(pred_rank[i] - pred_rank[j]) == np.sign(de[i] - de[j]))
    disc = n_pairs - agree
    return (agree - disc) / n_pairs, n_pairs


def mean_reliable_tau(
    answers: dict[str, list[str]],
    meta: dict[str, dict[str, Any]],
    floor: float,
) -> float:
    """Mean reliable-pair Kendall τ over molecules at the given |ΔE| floor."""
    taus: list[float] = []
    for mol, ranking in answers.items():
        if mol not in meta:
            continue
        letter_de = meta[mol]["letter_to_deltaE_kcal_mol"]
        letters = sorted(letter_de.keys(), key=label_sort_key)
        ranking = [str(x) for x in ranking]
        if set(ranking) != set(letters) or len(ranking) != len(letters):
            continue
        de = np.array([letter_de[L] for L in letters], dtype=np.float64)
        pred_rank = np.array([ranking.index(L) for L in letters], dtype=np.float64)
        tau, n_pairs = reliable_tau(de, pred_rank, floor)
        if n_pairs > 0 and np.isfinite(tau):
            taus.append(float(tau))
    return float(np.mean(taus)) if taus else float("nan")


def score_molecule(ranking: list[str], letter_de: dict[str, float],
                   floors: list[float]) -> dict[str, Any]:
    n = len(letter_de)
    letters = sorted(letter_de.keys(), key=label_sort_key)
    if set(ranking) != set(letters) or len(ranking) != n:
        return {"valid": False}

    de = np.array([letter_de[L] for L in letters], dtype=np.float64)
    pred_rank = np.array([ranking.index(L) for L in letters], dtype=np.float64)

    true_rank = np.argsort(np.argsort(de))
    tau_full = float(kendalltau(true_rank, pred_rank).statistic)

    row: dict[str, Any] = {"valid": True, "tau_full": tau_full}
    for floor in floors:
        tau, npairs = reliable_tau(de, pred_rank, floor)
        row[f"tau_f{floor}"] = tau
        row[f"n_f{floor}"] = npairs

    min_energy = float(de.min())
    ref_mins = {letters[i] for i, energy in enumerate(de) if energy == min_energy}
    row["top1"] = 1.0 if ranking[0] in ref_mins else 0.0
    row["min5"] = 1.0 if any(label in ref_mins for label in ranking[:5]) else 0.0

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--output-tag", type=str, required=True)
    ap.add_argument("--floors", type=float, nargs="+", default=DEFAULT_FLOORS)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write report to file (also prints to stdout)")
    add_baseline_arguments(ap)
    args = ap.parse_args()

    bundle = args.bundle_dir.resolve()
    meta_path = bundle / "prompt_metadata.jsonl"
    ans_path = bundle / f"answers_{args.output_tag}.jsonl"
    if not meta_path.is_file():
        print(f"Missing {meta_path}", file=sys.stderr)
        return 1
    if not ans_path.is_file():
        print(f"Missing {ans_path}", file=sys.stderr)
        return 1

    override = load_deltae_override(resolve_deltae_json(args.baseline, args.deltae_json))
    meta = augment_meta(
        load_metadata_index(meta_path),
        override,
        strict=override is not None,
    )
    answers = load_answers(ans_path)
    floors = sorted(set(args.floors))

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for mol, ranking in sorted(answers.items()):
        if mol not in meta:
            print(f"Warning: {mol!r} not in metadata, skip", file=sys.stderr)
            n_skipped += 1
            continue
        row = score_molecule(ranking, meta[mol]["letter_to_deltaE_kcal_mol"], floors)
        if not row["valid"]:
            print(f"Warning: {mol!r} not a valid permutation, skip", file=sys.stderr)
            n_skipped += 1
            continue
        row["mol"] = mol
        rows.append(row)

    if not rows:
        print("No valid answers.", file=sys.stderr)
        return 1

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], float)]
        return float(np.mean(vals)) if vals else float("nan")

    header = ["molecule", "tau_full", "top1"] + [f"tau_f{f}" for f in floors]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(
            [r["mol"], f"{r['tau_full']:.3f}", f"{r['top1']:.3f}"] +
            [f"{r[f'tau_f{f}']:.3f}" for f in floors]
        ))
    lines.append("")
    lines.append(f"MEAN across {len(rows)} molecules:")
    lines.append(f"  full tau          : {mean('tau_full'):.3f}")
    lines.append(f"  top-1 recovery    : {mean('top1'):.3f}")
    lines.append("  robustness curve (reliable-pair tau at floor):")
    for f in floors:
        lines.append(f"    floor {f:>4} : {mean(f'tau_f{f}'):.3f}  "
                     f"(mean {np.mean([r[f'n_f{f}'] for r in rows]):.0f} scored pairs of 435)")
    if n_skipped:
        lines.append(f"  (skipped {n_skipped} invalid/unknown molecules)")
    lines.append("")
    lines.append("# Notes: reliable-pair tau excludes pairs with |dE| < floor (reference unreliable there,"
                 " PNO flip <0.02% at floor 1.0).")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
