"""Pilot conformer pipeline: RDKit → xTB → CREST → ORCA r2SCAN-3c → prompts.

Processes llm_pp and llm_pm sequentially. Ollama benchmark is run separately.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors

RANDOM_SEED = 42
HARTREE_TO_KCAL = 627.509474063

_ORCA = Path(os.environ.get("ORCA_HOME", "/path/to/orca_6.1.1"))
if (_ORCA / "orca").is_file():
    os.environ["PATH"] = f"{_ORCA}:{os.environ.get('PATH', '')}"

N_THREADS = 16
SUBPROCESS_ENV = os.environ.copy()
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    SUBPROCESS_ENV[var] = str(N_THREADS)

SMILES_BY_PILOT = {
    "llm_pp": "C[C@@H](O)/C=C/[C@@H](C)OO",
    "llm_pm": r"CC(=O)OC1\C=C(C)/CCC2OC2(C)CC3OC(=O)C(=C)C13",
}
PILOTS = ("llm_pp", "llm_pm")

N_EMBED = 50
N_XTB_SEEDS = 10
TOP_N_EXPORT = 30
CHARGE = 0
UHF = 0
DEDUPE_RMSD = 0.1

REPO = Path(__file__).resolve().parent
ORCA_SCRIPT = Path(
    os.environ.get("ORCA_RUNNER", REPO / "orca_r2scan-3c" / "run_orca_local.sh")
)
BUILD_PROMPTS_SCRIPT = REPO / "prompts" / "build_prompts.py"


def _rel_path(path) -> str | None:
    """Repo-relative path when possible; basename otherwise (machine-independent manifests)."""
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return p.name


PROMPT_BUNDLES = (
    (42, "prompt_30x30_NEW"),
    (43, "prompt_30x30_NEW_43"),
    (44, "prompt_30x30_NEW_44"),
)


PILOT_DATA_ROOT = REPO / "geom-qm9"


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rdBase.SeedRandomNumberGenerator(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"Seeds fixed: random/numpy/RDKit SeedRandomNumberGenerator={seed}")


def mol_id_from_smiles(smiles: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in smiles)


def mol_conf_to_xyz(m: Chem.Mol, conf_id: int, comment: str = "") -> str:
    conf = m.GetConformer(conf_id)
    lines = [str(m.GetNumAtoms()), comment or "geom"]
    for atom in m.GetAtoms():
        x, y, z = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2} {x:12.8f} {y:12.8f} {z:12.8f}")
    return "\n".join(lines) + "\n"


def parse_xtb_energy(xtb_out: Path) -> float:
    text = xtb_out.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"TOTAL ENERGY\s+([-\d.]+)", text)
    if not m:
        raise ValueError(f"Could not parse xTB energy from {xtb_out}")
    return float(m.group(1))


def run_xtb_opt(xyz_path: Path, run_dir: Path) -> Path:
    cmd = [
        "xtb",
        xyz_path.name,
        "--gfn",
        "2",
        "--opt",
        "tight",
        "--chrg",
        str(CHARGE),
        "--uhf",
        str(UHF),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, env=SUBPROCESS_ENV, check=False)
    (run_dir / "xtb.out").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"xTB failed in {run_dir}:\n{proc.stdout}\n{proc.stderr}")
    opt = run_dir / "xtbopt.xyz"
    if not opt.is_file():
        raise FileNotFoundError(f"No xtbopt.xyz in {run_dir}")
    return opt


def run_crest(xyz_path: Path, run_dir: Path) -> Path:
    cmd = [
        "crest",
        xyz_path.name,
        "--gfn2",
        "--chrg",
        str(CHARGE),
        "--uhf",
        str(UHF),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, env=SUBPROCESS_ENV, check=False)
    (run_dir / "crest.out").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"CREST failed in {run_dir}:\n{proc.stdout}\n{proc.stderr}")
    ens = run_dir / "crest_conformers.xyz"
    if not ens.is_file():
        raise FileNotFoundError(f"No crest_conformers.xyz in {run_dir}")
    return ens


def parse_crest_multi_xyz(path: Path) -> list[tuple[float, tuple[str, ...]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[tuple[float, tuple[str, ...]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        e_hartree = float(lines[i + 1].strip())
        body = tuple(lines[i + 2 : i + 2 + n_atoms])
        if len(body) != n_atoms:
            raise ValueError(f"Truncated frame at line {i + 1} in {path}")
        out.append((e_hartree, body))
        i += 2 + n_atoms
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def embed_and_dedupe(mol: Chem.Mol, *, skip_mmff: bool) -> list[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.pruneRmsThresh = 0.01
    params.maxAttempts = 5
    params.useRandomCoords = False

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_EMBED, params=params))
    if not conf_ids:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=500, params=params))
    if not conf_ids:
        raise RuntimeError("RDKit embedding failed")
    print(f"embedded {len(conf_ids)} conformers (ETKDGv3, randomSeed={RANDOM_SEED})")

    energies: list[tuple[int, float]] = []
    for cid in sorted(conf_ids):
        e = 0.0
        if not skip_mmff:
            res = AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            if res == 0:
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                e = float(ff.CalcEnergy()) if ff else 0.0
        energies.append((cid, e))
    energies.sort(key=lambda t: (t[1], t[0]))

    kept: list[int] = []
    for cid, _ in energies:
        if not kept:
            kept.append(cid)
            continue
        if any(AllChem.GetBestRMS(mol, mol, k, cid) < DEDUPE_RMSD for k in kept):
            continue
        kept.append(cid)

    seed_ids = kept[:N_XTB_SEEDS]
    print(f"after dedupe: {len(kept)} unique; xTB seeds (conf_id order): {seed_ids}")
    return seed_ids


def run_pilot(pilot_root: str, smiles: str) -> tuple[Path, int]:
    seed_everything(RANDOM_SEED)

    mol_id = mol_id_from_smiles(smiles)
    work_dir = PILOT_DATA_ROOT / "_work" / mol_id
    out_dir = PILOT_DATA_ROOT / "output_30_confs" / mol_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    skip_mmff = "/" in smiles or "\\" in smiles
    print(f"\n{'=' * 60}")
    print(f"PILOT_ROOT: {pilot_root}")
    print(f"SMILES: {smiles}")
    print(f"heavy atoms: {mol.GetNumHeavyAtoms()}, rot bonds: {rdMolDescriptors.CalcNumRotatableBonds(mol)}")
    print(f"skip MMFF (E/Z in SMILES): {skip_mmff}")
    print(f"WORK_DIR: {work_dir}")
    print(f"OUT_DIR: {out_dir}")

    seed_ids = embed_and_dedupe(mol, skip_mmff=skip_mmff)

    xtb_log: list[dict] = []
    for rank, cid in enumerate(seed_ids):
        seed_dir = work_dir / f"xtb_seed_{rank}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_xyz = seed_dir / "seed.xyz"
        seed_xyz.write_text(mol_conf_to_xyz(mol, cid, f"embed conf {cid} rank {rank}"), encoding="utf-8")
        opt = run_xtb_opt(seed_xyz, seed_dir)
        e = parse_xtb_energy(seed_dir / "xtb.out")
        xtb_log.append({"rank": rank, "conf_id": cid, "energy_hartree": e, "xtbopt": _rel_path(opt)})
        print(f"  seed {rank} conf={cid}  E={e:.12f} Eh")

    best_rank = min(xtb_log, key=lambda row: (row["energy_hartree"], row["conf_id"]))
    best_e = best_rank["energy_hartree"]

    crest_input = work_dir / "crest" / "crest_input.xyz"
    crest_input.parent.mkdir(parents=True, exist_ok=True)
    crest_input.write_text(Path(best_rank["xtbopt"]).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nCREST input: conf_id={best_rank['conf_id']}  E={best_e:.12f} Eh  -> {crest_input}")

    crest_xyz = run_crest(crest_input, work_dir / "crest")
    print(f"CREST ensemble: {crest_xyz}")

    conformers_hartree = parse_crest_multi_xyz(crest_xyz)
    e_min = conformers_hartree[0][0]
    conformers = [((e_h - e_min) * HARTREE_TO_KCAL, body) for e_h, body in conformers_hartree]

    n_export = min(TOP_N_EXPORT, len(conformers))
    print(f"CREST returned {len(conformers)} conformers; exporting {n_export}")
    if n_export:
        print(f"ΔE range: 0.000 – {conformers[n_export - 1][0]:.3f} kcal/mol (xTB)")

    for i, (rel_e_kcal, body) in enumerate(conformers[:n_export]):
        out_path = out_dir / f"{mol_id}_conf_{i}.xyz"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"{len(body)}\n")
            f.write(f"Energy: {rel_e_kcal:.6f}\n")
            f.write("\n".join(body) + "\n")

    manifest = {
        "pilot_root": pilot_root,
        "smiles": smiles,
        "mol_id": mol_id,
        "reproducibility": {
            "conda_env": "llm_benchmark",
            "random_seed": RANDOM_SEED,
            "rdkit_etkdg_randomSeed": RANDOM_SEED,
            "rdkit_SeedRandomNumberGenerator": RANDOM_SEED,
            "subprocess_OMP_NUM_THREADS": str(N_THREADS),
            "xtb": _rel_path(shutil.which("xtb")),
            "crest": _rel_path(shutil.which("crest")),
            "orca": _rel_path(shutil.which("orca")),
            "crest_note": "CREST MTD may differ in ensemble size; xTB+RDKit path is deterministic",
        },
        "n_embed": N_EMBED,
        "skip_mmff": skip_mmff,
        "seed_conf_ids": seed_ids,
        "crest_input_conf_id": best_rank["conf_id"],
        "crest_input_energy_hartree": best_e,
        "energy_units": "kcal/mol (ΔE vs xTB minimum in CREST ensemble)",
        "e_min_hartree": e_min,
        "n_crest_conformers": len(conformers),
        "n_exported": n_export,
        "output_dir": _rel_path(out_dir),
        "xtb_seeds": xtb_log,
    }
    manifest_path = out_dir.parent / f"{mol_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")
    print(f"XYZ files -> {out_dir}/")
    return manifest_path, n_export


def run_orca(pilot_root: str) -> Path:
    """Run r2SCAN-3c ORCA via orca_r2scan-3c/run_orca_local.sh (same as manual env vars)."""
    conformers_dir = PILOT_DATA_ROOT / "output_30_confs"
    out_dir = REPO / "orca_r2scan-3c" / pilot_root
    deltae_json = out_dir / "deltaE_r2scan3c_kcal_mol.json"
    if not ORCA_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing ORCA runner: {ORCA_SCRIPT}")

    env = os.environ.copy()
    env["CONFORMERS_DIR"] = str(conformers_dir)
    env["OUT_DIR"] = str(out_dir)

    print(f"\n{'=' * 60}")
    print(f"ORCA: CONFORMERS_DIR={conformers_dir}")
    print(f"ORCA: OUT_DIR={out_dir}")
    subprocess.run([str(ORCA_SCRIPT)], cwd=REPO, env=env, check=True)
    print(f"ORCA deltaE -> {deltae_json}")
    return deltae_json


def build_prompts_for_pilot(pilot_root: str, n_confs: int) -> None:
    data_dir = PILOT_DATA_ROOT / "output_30_confs"
    if not BUILD_PROMPTS_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing {BUILD_PROMPTS_SCRIPT}")

    print(f"\n{'=' * 60}")
    print(f"Building prompts ({n_confs} conformers) from {data_dir}")
    for seed, bundle_name in PROMPT_BUNDLES:
        out_dir = PILOT_DATA_ROOT / bundle_name
        print(f"  seed {seed} -> {out_dir}")
        subprocess.run(
            [
                sys.executable,
                str(BUILD_PROMPTS_SCRIPT),
                "--data-dir",
                str(data_dir),
                "--n-confs",
                str(n_confs),
                "--out-dir",
                str(out_dir),
                "--seed",
                str(seed),
            ],
            cwd=REPO,
            check=True,
        )


def main() -> int:
    for exe in ("xtb", "crest", "orca"):
        if shutil.which(exe) is None:
            raise RuntimeError(
                f"{exe} not on PATH. Activate conda env `llm_benchmark` "
                f"(xtb/crest: conda-forge; orca: set ORCA_HOME, e.g. /path/to/orca_6.1.1)."
            )

    print("python:", sys.version.split()[0])
    print("rdkit:", rdBase.rdkitVersion)
    print("xtb:", shutil.which("xtb"))
    print("crest:", shutil.which("crest"))
    print("orca:", shutil.which("orca"))
    subprocess.run(["xtb", "--version"], env=SUBPROCESS_ENV, check=False)

    for pilot_root in PILOTS:
        _, n_export = run_pilot(pilot_root, SMILES_BY_PILOT[pilot_root])
        run_orca(pilot_root)
        build_prompts_for_pilot(pilot_root, n_export)

    print(f"\nDone — {len(PILOTS)} pilot(s): {', '.join(PILOTS)}")
    print("Next: run Ollama benchmark on prompt_30x30_NEW{,_43,_44}/ under each pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
