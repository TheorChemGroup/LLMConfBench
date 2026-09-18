#!/usr/bin/env python3

"""Parse DLPNO-CCSD(T) ORCA .out files into deltaE_dlpno_ccsdt_kcal_mol.json (10-molecule subset)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ENERGY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)"),
    re.compile(r"DLPNO-CCSD\(T\) total energy\s*[:\s=]*\s*([-\d.]+)", re.IGNORECASE),
    re.compile(r"The final energy is\s+([-\d.]+)"),
    re.compile(r"Total\s+energy\s*[:\s=]*\s*([-\d.]+)", re.IGNORECASE),
    re.compile(r"Electronic\s+energy\s*[:\s=]*\s*([-\d.]+)", re.IGNORECASE),
]

KCAL = 627.509469


def parse_sp_energy(out_text: str) -> float | None:
    for pat in ENERGY_PATTERNS:
        m = pat.search(out_text)
        if m:
            return float(m.group(1))
    return None


def parse_all_energies(out_dir: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for mol_dir in sorted(out_dir.iterdir()):
        if not mol_dir.is_dir():
            continue
        mol = mol_dir.name
        mol_energies: dict[int, float] = {}
        for out_path in sorted(mol_dir.glob(f"{mol}_conf_*.out")):
            m = re.search(r"_conf_(\d+)\.out$", out_path.name)
            if not m:
                continue
            ci = int(m.group(1))
            try:
                text = out_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            energy = parse_sp_energy(text)
            if energy is not None:
                mol_energies[ci] = energy
        if mol_energies:
            result[mol] = mol_energies
    return result


def sp_to_deltaE(mol_energies: dict[int, float]) -> dict[int, float]:
    min_e = min(mol_energies.values())
    return {ci: (e - min_e) * KCAL for ci, e in mol_energies.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "orca_dlpno_ccsdt")
    ap.add_argument("--deltae-json", type=Path, default=HERE / "deltaE_dlpno_ccsdt_kcal_mol.json")
    args = ap.parse_args()

    energies = parse_all_energies(args.out_dir)
    if not energies:
        print("No ORCA .out files found; nothing to parse.", file=sys.stderr)
        return 1
    print(f"Parsed DLPNO-CCSD(T) energies for {len(energies)} molecules")

    deltaE_by_mol = {mol: sp_to_deltaE(e) for mol, e in energies.items()}
    args.deltae_json.write_text(
        json.dumps(
            {
                mol: {str(ci): round(de, 4) for ci, de in de_map.items()}
                for mol, de_map in deltaE_by_mol.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote deltaE baseline -> {args.deltae_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
