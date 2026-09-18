#!/usr/bin/env python3

"""Pick a 10-molecule CCSD(T) subset spanning the r2SCAN-3c ΔE range."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltae-json", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "orca_r2scan-3c" / "deltaE_r2scan3c_kcal_mol.json")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "subset_molecules.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.deltae_json.read_text(encoding="utf-8"))
    spans: list[tuple[str, float]] = []
    for mol, de in data.items():
        vals = [float(v) for v in de.values()]
        spans.append((mol, float(np.max(vals) - np.min(vals))))
    spans.sort(key=lambda x: x[1])

    n = min(args.size, len(spans))
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    if n >= len(spans):
        chosen = [m for m, _ in spans]
    else:
        idx = np.linspace(0, len(spans) - 1, n).astype(int)
        if args.seed is not None:
            idx = idx + rng.integers(0, max(len(spans) // n, 1), size=n)
            idx = np.minimum(idx, len(spans) - 1)
        chosen = [spans[i][0] for i in idx]

    args.out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print("subset molecules (name, r2scan span kcal/mol):")
    by_span = {m: s for m, s in spans}
    for m in chosen:
        print(f"  {m:<45s} {by_span[m]:8.2f}")
    print(f"wrote -> {args.out} ({len(chosen)} molecules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
