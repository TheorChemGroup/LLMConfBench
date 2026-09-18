from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_meta(bundle_dir: Path) -> list[dict]:
    out: list[dict] = []
    with (bundle_dir / "prompt_metadata.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def min_at_k_random(n: int, m_positives: int, k: int) -> float:
    """P(at least one positive in top-k) under uniform random ranking."""
    if k >= n:
        return 1.0
    if m_positives <= 0:
        return 0.0
    miss = math.comb(n - m_positives, k) / math.comb(n, k)
    return 1.0 - miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir", type=Path)
    args = ap.parse_args()

    meta_rows = load_meta(args.bundle_dir)

    print(f"bundle: {args.bundle_dir}   molecules: {len(meta_rows)}")
    print()
    print("Kendall tau   : E[tau] = 0.000  (by symmetry, for any random permutation)")
    print()

    ks = (1, 3, 5, 10)
    acc = {k: 0.0 for k in ks}

    print(
        f"{'molecule':<30} {'N':>3} {'m+':>3}  "
        + "  ".join(f"min@{k}" for k in ks)
    )
    print("-" * 100)

    for row in meta_rows:
        dE: dict[str, float] = row["letter_to_deltaE_kcal_mol"]
        n = row["n_conformers"]
        min_e = min(dE.values())
        m = sum(1 for v in dE.values() if v == min_e)

        min_k_vals = {k: min_at_k_random(n, m, k) for k in ks}

        for k, v in min_k_vals.items():
            acc[k] += v

        print(
            f"{row['molecule'][:30]:<30} {n:>3} {m:>3}  "
            + "  ".join(f"{min_k_vals[k]:.3f}" for k in ks)
        )

    print("-" * 100)
    n_mol = len(meta_rows)
    print(
        f"{'AVERAGE':<30} {'':>3} {'':>3}  "
        + "  ".join(f"{acc[k] / n_mol:.3f}" for k in ks)
    )


if __name__ == "__main__":
    main()
