#!/usr/bin/env python3

"""Write r2SCAN-3c ORCA inputs for every conformer under geom-qm9/output_30_confs (30 GEOM-QM9 + 2 de-novo molecules)."""

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
# r2SCAN-3c single-point energy, produced for the conformer benchmark
! r2SCAN-3c SP
! ENGRAD
%pal nprocs {nprocs} end
"""


def generate_inputs(conformers_dir: Path, out_dir: Path, nprocs: int) -> int:
    if not conformers_dir.is_dir():
        raise SystemExit(f"Conformers dir not found: {conformers_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for mol_dir in sorted(conformers_dir.iterdir()):
        if not mol_dir.is_dir() or mol_dir.name.startswith("."):
            continue
        mol = mol_dir.name
        mol_out = out_dir / mol
        mol_out.mkdir(parents=True, exist_ok=True)
        for xyz_path in sorted(mol_dir.glob(f"{mol}_conf_*.xyz")):
            inp_name = xyz_path.with_suffix(".inp").name
            inp_text = ORCA_HEADER.format(nprocs=nprocs) + "\n* xyzfile 0 1 " + xyz_path.name + "\n"
            (mol_out / inp_name).write_text(inp_text, encoding="utf-8")
            shutil.copy2(xyz_path, mol_out / xyz_path.name)
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conformers-dir", type=Path, default=geom_conformers_dir())
    ap.add_argument("--out-dir", type=Path, default=HERE / "orca_r2scan3c")
    ap.add_argument("--nprocs", type=int, default=64,
                    help="Number of ORCA threads per job (%%pal nprocs).")
    args = ap.parse_args()

    n = generate_inputs(args.conformers_dir, args.out_dir, args.nprocs)
    print(f"Wrote {n} ORCA r2SCAN-3c SP inputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
