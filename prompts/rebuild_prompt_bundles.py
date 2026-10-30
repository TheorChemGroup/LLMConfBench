"""Rebuild the committed prompt bundles for all three seeds.

    python prompts/rebuild_prompt_bundles.py --out-dir geom-qm9

Each bundle (``prompt_30x30_NEW``, ``..._43``, ``..._44``) is reproduced the way
it was originally generated, which is required for byte-exact output:

* one ``build_prompts.py`` run over the **original 30 GEOM-QM9 molecules**
  (sorted, *without* the two de-novo pilots), and
* one **separate run per pilot**.

``build_prompts.py`` draws from a single ``random.Random(seed)`` per invocation,
so a molecule's label permutation and block order depend on the set and order of
the molecules processed before it.  A single run over today's ``output_30_confs``
(32 folders) or over the 27-molecule cohort yields different prompts.

The bundle's ``prompt_metadata.jsonl`` is the merge of the GEOM run (restricted
to the 25-molecule GEOM cohort) and the two pilot rows, matching the committed
file.  Rebuilding overwrites the bundles, so do it only when you intend to re-run
all models.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BUILD = Path(__file__).resolve().parent / "build_prompts.py"

MACROCYCLE = "CC__O_OC1_C_C_C__CCC2OC2_C_CC3OC__O_C__C_C13"
PEROXIDE = "C_C__H__O__C_C__C__H__C_OO"
PILOTS = (MACROCYCLE, PEROXIDE)

EXCLUDED = frozenset(
    {
        "CCCCCNC_O",
        "CCC_C__H__C_COC_O",
        "CC_C_H_1CC_C____C__O_C1",
        "CC__O__C__H__C_CC1CC1",
        "C_C_H_1N_C__1_CO_CC_O",
    }
)
SEED_BUNDLES = (
    (42, "prompt_30x30_NEW"),
    (43, "prompt_30x30_NEW_43"),
    (44, "prompt_30x30_NEW_44"),
)


def geom30(data_dir: Path) -> list[str]:
    """The original 30 GEOM-QM9 folder names, sorted (pilots excluded)."""
    return [
        p.name
        for p in sorted(data_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".") and p.name not in PILOTS
    ]


def run_build(molecules: list[str], seed: int, out_dir: Path, data_dir: Path) -> list[dict]:
    """Run build_prompts.py for one molecule set; return its metadata rows."""
    subprocess.run(
        [
            sys.executable,
            str(_BUILD),
            "--data-dir", str(data_dir),
            "--n-confs", "30",
            "--seed", str(seed),
            "--out-dir", str(out_dir),
            "--molecules", *molecules,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return [
        json.loads(line)
        for line in (out_dir / "prompt_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_bundle(seed: int, name: str, out_root: Path, data_dir: Path, tmp: Path) -> None:
    geom = geom30(data_dir)
    cohort = [m for m in geom if m not in EXCLUDED]

    geom_dir = tmp / f"geom_{seed}"
    geom_rows = run_build(geom, seed, geom_dir, data_dir)

    pilot_rows: list[dict] = []
    pilot_dirs: list[Path] = []
    for i, pilot in enumerate(PILOTS):
        pdir = tmp / f"pilot_{seed}_{i}"
        pilot_rows += run_build([pilot], seed, pdir, data_dir)
        pilot_dirs.append(pdir)

    bundle = out_root / name
    bundle.mkdir(parents=True, exist_ok=True)


    for src_dir in (geom_dir, *pilot_dirs):
        for txt in src_dir.glob("*.txt"):
            shutil.copyfile(txt, bundle / txt.name)


    rows = [r for r in geom_rows if r["molecule"] in cohort] + pilot_rows
    try:
        pilot_prefix = bundle.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        pilot_prefix = name
    for row in geom_rows:
        row["txt_path"] = f"{name}/{Path(row['txt_path']).name}"
    for row in pilot_rows:
        row["txt_path"] = f"{pilot_prefix}/{Path(row['txt_path']).name}"
    (bundle / "prompt_metadata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_REPO / "geom-qm9" / "output_30_confs")
    ap.add_argument("--out-dir", type=Path, default=_REPO / "geom-qm9")
    args = ap.parse_args()

    for seed, name in SEED_BUNDLES:
        with tempfile.TemporaryDirectory() as td:
            build_bundle(seed, name, args.out_dir, args.data_dir, Path(td))
        print(f"seed {seed}: rebuilt {args.out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
