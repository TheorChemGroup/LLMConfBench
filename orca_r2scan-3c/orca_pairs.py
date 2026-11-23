"""Shared readers and pair selection for the ORCA cross-checks.

The three comparison scripts (``compare_baselines_same_pairs.py``,
``recompute_accuracy_r2scan3c.py``, ``orca_ccsdt/compare_baselines_subset.py``)
score the same LLM rankings against reference energies, so the file readers and
the per-mode pair selection live here once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEED_DIRS: list[tuple[str, Path]] = [
    ("42", Path("prompt_30x30_NEW")),
    ("43", Path("prompt_30x30_NEW_43")),
    ("44", Path("prompt_30x30_NEW_44")),
]

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
        if mol is not None and isinstance(rnk, list):
            out[str(mol)] = [str(x) for x in rnk]
    return out


def label_to_conf_ranking(row: dict[str, Any], labels: list[str]) -> list[int] | None:
    """Public labels -> conformer indices, or None if the ranking is malformed."""
    pub_map = row["public_label_to_conf_index"]
    if len(labels) != len(pub_map):
        return None
    try:
        conf = [int(pub_map[lab]) for lab in labels]
    except KeyError:
        return None
    if len(set(conf)) != len(conf):
        return None
    return conf


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
    """Subset of pairs for a difficulty mode: normal / hard / extreme_hard."""
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
