#!/usr/bin/env python3

"""Write DLPNO-CCSD(T) ORCA inputs for the 10-molecule subset of the r2SCAN-3c DFT set."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import geom_conformers_dir

HERE = Path(__file__).resolve().parent

ORCA_HEADER = """\
# DLPNO-CCSD(T)/def2-TZVP single-point energy (subset validation)
! DLPNO-CCSD(T) def2-TZVP RIJCOSX TightPNO cc-pVTZ/C SP
%pal nprocs {nprocs} end
%maxcore {memory_mb}
"""


def load_subset(subset_path: Path) -> list[str]:
    if not subset_path.is_file():
        raise SystemExit(f"Subset file not found: {subset_path} "
                         "(run select_subset.py first)")
    return [
        ln.strip()
        for ln in subset_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def generate_inputs(subset: list[str], conformers_dir: Path, out_dir: Path,
                    nprocs: int, memory_mb: int) -> int:
    if not conformers_dir.is_dir():
        raise SystemExit(f"Conformers dir not found: {conformers_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    missing = 0
    for mol in subset:
        mol_dir = conformers_dir / mol
        if not mol_dir.is_dir():
            print(f"WARNING: no conformer dir for {mol!r}, skip", file=sys.stderr)
            missing += 1
            continue
        mol_out = out_dir / mol
        mol_out.mkdir(parents=True, exist_ok=True)
        for xyz_path in sorted(mol_dir.glob(f"{mol}_conf_*.xyz")):
            inp_name = xyz_path.with_suffix(".inp").name
            inp_text = ORCA_HEADER.format(nprocs=nprocs, memory_mb=memory_mb) \
                + "\n* xyzfile 0 1 " + xyz_path.name + "\n"
            (mol_out / inp_name).write_text(inp_text, encoding="utf-8")
            shutil.copy2(xyz_path, mol_out / xyz_path.name)
            count += 1
    if missing:
        print(f"WARNING: {missing} molecules from the subset had no conformer dir", file=sys.stderr)
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=Path, default=HERE / "subset_molecules.txt")
    ap.add_argument("--conformers-dir", type=Path, default=geom_conformers_dir())
    ap.add_argument("--out-dir", type=Path, default=HERE / "orca_dlpno_ccsdt")
    ap.add_argument("--nprocs", type=int, default=16,
                    help="ORCA threads per job (%%pal nprocs).")
    ap.add_argument("--memory-mb", type=int, default=4000,
                    help="Memory per ORCA thread (%%maxcore, MB).")
    args = ap.parse_args()

    subset = load_subset(args.subset)
    n = generate_inputs(subset, args.conformers_dir, args.out_dir,
                        args.nprocs, args.memory_mb)
    print(f"Wrote {n} DLPNO-CCSD(T) SP inputs ({len(subset)} molecules) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
