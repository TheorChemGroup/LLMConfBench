#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as t_dist

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from config.models_config import (
    PILOT_MOLECULE_IDS,
    add_baseline_arguments,
    geom_seed_paths,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import (
    augment_meta,
    kendall_for_molecule,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import load_valid_answers


def compute_per_molecule(
    answers: dict[str, list[str]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for mol, rnk in answers.items():
        if mol not in meta:
            continue
        de = meta[mol]["letter_to_deltaE_kcal_mol"]
        keys = set(de.keys())
        rnk_str = [str(x) for x in rnk]
        if set(rnk_str) != keys or len(rnk_str) != len(keys):
            continue

        tau, _ = kendall_for_molecule(de, rnk_str)
        result[mol] = tau
    return result


def mean_std_ci(values: np.ndarray) -> tuple[float, float, float, float, float]:
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0, 0.0, mean, mean
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(t_dist.ppf(0.975, df=n - 1))
    ci_lo = mean - t_crit * se
    ci_hi = mean + t_crit * se
    return mean, std, se, ci_lo, ci_hi


def has_three_complete_seeds(
    answers_filename: str,
    seed_dirs: list[Path] | None = None,
    deltae_override: dict | None = None,
) -> bool:
    dirs = list(seed_dirs) if seed_dirs is not None else geom_seed_paths()
    for seed_dir in dirs:
        meta_path = seed_dir / "prompt_metadata.jsonl"
        if not meta_path.is_file():
            return False
        meta = augment_meta(
            load_metadata_index(meta_path),
            deltae_override,
            strict=deltae_override is not None,
        )
        answers = load_valid_answers(seed_dir / answers_filename, meta)
        required = set(meta) - PILOT_MOLECULE_IDS
        if not required.issubset(answers):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle-dirs", type=Path, nargs="+", default=None,
        help="Prompt bundle directories (one per seed; default from models.yaml)",
    )
    ap.add_argument(
        "--output-tag", type=str, required=True,
        help="Answers file tag: reads answers_{tag}.jsonl from each bundle",
    )
    add_baseline_arguments(ap)
    args = ap.parse_args()

    tag = args.output_tag.strip()
    bundle_dirs = args.bundle_dirs if args.bundle_dirs is not None else geom_seed_paths()
    override = load_deltae_override(resolve_deltae_json(args.baseline, args.deltae_json))

    all_taus: dict[str, list[float]] = {}
    n_seeds_ok = 0

    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "prompt_metadata.jsonl"
        if not meta_path.is_file():
            print(f"Warning: {meta_path} not found, skipping", file=sys.stderr)
            continue
        meta = augment_meta(
            load_metadata_index(meta_path),
            override,
            strict=override is not None,
        )

        ans_path = bundle_dir / f"answers_{tag}.jsonl"
        if not ans_path.is_file():
            print(f"Warning: {ans_path} not found, skipping", file=sys.stderr)
            continue
        answers = load_valid_answers(ans_path, meta)
        if not answers:
            print(f"Warning: no valid answers in {ans_path}", file=sys.stderr)
            continue
        n_seeds_ok += 1
        per_mol = compute_per_molecule(answers, meta)
        for mol, tau in per_mol.items():
            all_taus.setdefault(mol, []).append(tau)

    if n_seeds_ok == 0:
        print("No valid answers found across all bundles.", file=sys.stderr)
        return 1

    all_molecules = sorted(all_taus.keys())
    n_mols = len(all_molecules)

    header = f"{'molecule':<30s} {'tau_mean':>8s} {'tau_std':>8s} {'tau_SE':>8s} {'tau_95CI':>20s}"
    print(header)
    print("-" * len(header))

    tau_means = []

    for mol in all_molecules:
        taus = np.array(all_taus[mol])
        t_m, t_s, t_se, t_lo, t_hi = mean_std_ci(taus)
        tau_means.append(t_m)
        print(
            f"{mol:<30s} {t_m:+8.3f} {t_s:8.3f} {t_se:8.3f} [{t_lo:+8.3f}, {t_hi:+8.3f}]"
        )

    tau_vec = np.array(tau_means)
    t_m, t_s, t_se, t_lo, t_hi = mean_std_ci(tau_vec)

    print("-" * len(header))
    print(
        f"{'AGGREGATE':<30s} {t_m:+8.3f} {t_s:8.3f} {t_se:8.3f} [{t_lo:+8.3f}, {t_hi:+8.3f}]"
    )
    print(f"\nn_molecules={n_mols}  n_seeds={n_seeds_ok}  CI=95% (t-distribution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
