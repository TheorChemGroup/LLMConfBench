#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import kendalltau


def load_deltae_override(path: Path | None) -> dict[str, dict[int, float]] | None:
    if path is None:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read alternate baseline JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid alternate baseline JSON: {path}") from exc
    return {mol: {int(k): float(v) for k, v in de.items()}
            for mol, de in raw.items()}


def _override_energy_map(
    meta_row: dict[str, Any],
    deltae_override: dict[str, dict[int, float]],
) -> dict[str, float] | None:
    mol = str(meta_row["molecule"])
    override = deltae_override.get(mol)
    pub = meta_row.get("public_label_to_conf_index", {})
    if not override or not pub:
        return None
    out: dict[str, float] = {}
    for lab, ci in pub.items():
        energy = override.get(int(ci))
        if energy is None:
            return None
        out[str(lab)] = float(energy)
    return out if len(out) == len(pub) else None


def energy_map_for(meta_row: dict[str, Any], deltae_override: dict[str, dict[int, float]] | None) -> dict[str, float]:
    if deltae_override is not None:
        out = _override_energy_map(meta_row, deltae_override)
        if out is not None:
            return out
    return meta_row["letter_to_deltaE_kcal_mol"]


def augment_meta(meta: dict[str, dict[str, Any]],
                 deltae_override: dict[str, dict[int, float]] | None,
                 *,
                 strict: bool = False) -> dict[str, dict[str, Any]]:
    if deltae_override is None:
        return meta
    out: dict[str, dict[str, Any]] = {}
    for mol, row in meta.items():
        de = _override_energy_map(row, deltae_override)
        if de is None:
            if strict:
                continue
            de = row["letter_to_deltaE_kcal_mol"]
        row = dict(row)
        row["letter_to_deltaE_kcal_mol"] = de
        out[mol] = row
    return out


def load_metadata_index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        m = row.get("molecule")
        if m is not None:
            out[str(m)] = row
    return out


def ideal_order_by_energy(letter_to_de: dict[str, float]) -> list[str]:

    def tie(L: str) -> tuple:
        try:
            return (0, int(L))
        except ValueError:
            return (1, L)

    return sorted(letter_to_de.keys(), key=lambda L: (letter_to_de[L], tie(L)))


def _label_sort_key(x: str) -> tuple:
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


label_sort_key = _label_sort_key


def ordered_labels_for_kendall(keys: list[str]) -> list[str]:
    return sorted(keys, key=_label_sort_key)


def kendall_for_molecule(
    letter_to_de: dict[str, float],
    pred_ranking: list[str],
) -> tuple[float, float]:
    pred_ranking = [str(x) for x in pred_ranking]
    letters = ordered_labels_for_kendall(list(letter_to_de.keys()))
    if set(pred_ranking) != set(letters):
        raise ValueError("Predicted ranking must be a permutation of metadata letters")
    ideal = ideal_order_by_energy(letter_to_de)
    true_ranks = [ideal.index(c) for c in letters]
    pred_ranks = [pred_ranking.index(c) for c in letters]
    res = kendalltau(true_ranks, pred_ranks)
    return float(res.statistic), float(res.pvalue)


def process_bundle(
    bundle_dir: Path,
) -> tuple[list[tuple[str, float, float]], float]:
    meta_path = bundle_dir / "prompt_metadata.jsonl"
    ans_path = bundle_dir / "answers.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not ans_path.is_file():
        raise FileNotFoundError(f"Missing {ans_path}")

    meta = load_metadata_index(meta_path)
    rows: list[tuple[str, float, float]] = []
    taus: list[float] = []

    for line in ans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        mol = str(obj["molecule"])
        ranking = obj["ranking"]
        if mol not in meta:
            raise ValueError(f"No metadata for molecule {mol!r} in {meta_path}")
        letter_de: dict[str, float] = meta[mol]["letter_to_deltaE_kcal_mol"]
        tau, pval = kendall_for_molecule(letter_de, ranking)
        rows.append((mol, tau, pval))
        taus.append(tau)

    mean_tau = sum(taus) / len(taus) if taus else float("nan")
    return rows, mean_tau


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kendall tau vs deltaE ordering (same files as nDCG pipeline)."
    )
    ap.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Folders each containing prompt_metadata.jsonl and answers.jsonl",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Write report (UTF-8); stdout unchanged",
    )
    args = ap.parse_args()

    chunks: list[str] = []
    for d in args.dirs:
        d = d.resolve()
        chunks.append(f"## {d}")
        rows, mean_tau = process_bundle(d)
        for mol, tau, pval in rows:
            chunks.append(f"{mol}\tKendall_tau={tau:.3f}\tp_value={pval:.3f}")
        chunks.append(f"mean Kendall tau: {mean_tau:.3f}  (n={len(rows)})")
        chunks.append("")

    text = "\n".join(chunks).rstrip() + "\n"

    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
