#!/usr/bin/env python3

"""Shared scoring, force-field rankings, and scatter helpers for manuscript plots."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import (
    MARKER_DEFAULT,
    MARKER_EDGEWIDTH,
    MARKER_REASONING,
    MARKER_SIZE,
    PILOT_MOLECULE_IDS,
    TAU_FLOOR,
    TAU_KEY,
    VENDOR_FAMILIES,
    color_for_display,
    default_models_table,
    family_for_display,
    family_legend_color,
    geom_seed_dirs,
    geom_seed_paths,
    is_reasoning_or_thinking,
    marker_for_display,
    models_in_group,
    mols_md_path,
    scatter_style_for_display,
)
from metrics.grouped_metrics import mean_std_ci
from metrics.kendall_tau_ranking import (
    augment_meta,
    kendall_for_molecule,
    load_metadata_index,
)
from metrics.option2_metrics import (
    load_valid_answers,
    mean_reliable_tau,
    score_molecule,
)
from viz.plot_style import (
    REF_GFNFF,
    REF_MMFF,
    REF_MMFF_LINESTYLE,
    REF_RANDOM,
    REF_UFF,
    plt,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from rdkit.Geometry import Point3D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

if RDKIT_AVAILABLE:
    from rdkit import RDLogger
    RDLogger.EnableLog('rdApp.*')
else:
    print("WARNING: RDKit is not installed; UFF rankings cannot be computed.", file=sys.stderr)

FF_CACHE_DIRNAME = ".ff_cache"
MMFF94_NAME = "MMFF94"

_FF_CACHE_TAGS = {
    "UFF": "uff_smiles",
    "GFN-FF": "gfnff",
    MMFF94_NAME: "mmff94_smiles",
}

_FF_SMILES_METHODS = frozenset({"UFF", MMFF94_NAME})
_FF_RANK_MEMO: dict[tuple[str, str], list[int]] = {}


_UFF_SMILES_OVERRIDE = {
    "C_C_H__C_O_CO_CH__NH_": "C[C@H](C=O)COC=N",
    "C_C___NH__OCCCCO": "CC(=N)OCCCCO",
}

from functools import lru_cache


def _sanitize_smiles_key(smi: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in smi)


@lru_cache(maxsize=1)
def _smiles_by_folder() -> dict[str, str]:
    """Folder name (sanitized SMILES) -> SMILES used for UFF topology."""
    import ast

    out: dict[str, str] = {}
    for smi in ast.literal_eval(mols_md_path().read_text(encoding="utf-8").strip()):
        key = _sanitize_smiles_key(smi)
        out[key] = _UFF_SMILES_OVERRIDE.get(key, smi)
    return out


def _xyz_fingerprint(folder_path: Path, *, smiles: str | None = None) -> str:
    parts: list[str] = []
    for path in sorted(folder_path.glob("*.xyz")):
        if re.search(r"conf_(\d+)\.xyz$", path.name) is None:
            continue
        st = path.stat()
        parts.append(f"{path.name}:{st.st_size}:{st.st_mtime_ns}")
    if smiles is not None:
        parts.append(f"smiles:{smiles}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _ff_cache_path(folder_path: Path, method: str) -> Path:
    tag = _FF_CACHE_TAGS[method]
    return folder_path.parent / FF_CACHE_DIRNAME / tag / f"{folder_path.name}.json"


def _load_cached_ranking(folder_path: Path, method: str) -> list[int] | None:
    key = (method, str(folder_path.resolve()))
    memo = _FF_RANK_MEMO.get(key)
    if memo is not None:
        return memo
    path = _ff_cache_path(folder_path, method)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 30:
        return None
    smiles = None
    if method in _FF_SMILES_METHODS:
        smiles = _smiles_by_folder().get(folder_path.name)
        if smiles is None or payload.get("smiles") != smiles:
            return None
    if payload.get("fingerprint") != _xyz_fingerprint(folder_path, smiles=smiles):
        return None
    ranking_i = [int(x) for x in ranking]
    _FF_RANK_MEMO[key] = ranking_i
    return ranking_i


def _store_cached_ranking(
    folder_path: Path,
    method: str,
    ranking: list[int],
    *,
    smiles: str | None = None,
) -> None:
    key = (method, str(folder_path.resolve()))
    _FF_RANK_MEMO[key] = ranking
    path = _ff_cache_path(folder_path, method)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "fingerprint": _xyz_fingerprint(folder_path, smiles=smiles),
            "ranking": ranking,
        }
        if smiles is not None:
            payload["smiles"] = smiles
            payload["topology"] = "smiles"
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _ranking_cached(
    folder_path: Path,
    method: str,
    compute_fn,
    *,
    smiles: str | None = None,
) -> list[int] | None:
    cached = _load_cached_ranking(folder_path, method)
    if cached is not None:
        return cached
    ranking = compute_fn(folder_path)
    if ranking is not None and len(ranking) == 30:
        _store_cached_ranking(folder_path, method, ranking, smiles=smiles)
    return ranking


def get_uff_ranking(folder_path: Path) -> list[int] | None:
    """UFF ranking of conf indices (lowest energy first). Cached on disk.

    Topology comes from MOLS.md SMILES (bond orders via AssignBondOrdersFromTemplate);
    geometries come from the exported XYZ files.
    """
    smiles = _smiles_by_folder().get(folder_path.name)
    return _ranking_cached(
        folder_path, "UFF", _compute_uff_ranking, smiles=smiles,
    )


def _indexed_xyz(folder_path: Path) -> list[tuple[int, Path]] | None:
    """Sorted [(conf_index, path)] for a folder; None unless exactly 30 confs."""
    indexed = []
    for p in folder_path.glob("*.xyz"):
        match = re.search(r"conf_(\d+)\.xyz$", p.name)
        if match:
            indexed.append((int(match.group(1)), p))
    indexed.sort(key=lambda x: x[0])
    if len(indexed) != 30:
        print(f"[{folder_path.name}] Error: found {len(indexed)} files, expected 30.")
        return None
    return indexed


def _build_template_with_conformers(
    folder_path: Path,
    method: str,
) -> tuple[Chem.Mol, dict[int, int]] | None:
    """SMILES-topology template with every conf_N.xyz added as a conformer.

    Returns (mol, {conformer_id: original conf index}); None on any failure.
    """
    if not RDKIT_AVAILABLE:
        print(f"[{folder_path.name}] Error: RDKit is not installed.")
        return None
    smiles = _smiles_by_folder().get(folder_path.name)
    if not smiles:
        print(f"[{folder_path.name}] Error: no SMILES in {mols_md_path()} for this folder.")
        return None
    indexed_files = _indexed_xyz(folder_path)
    if indexed_files is None:
        return None

    first_xyz_path = indexed_files[0][1]
    try:
        raw_mol = Chem.MolFromXYZFile(str(first_xyz_path))
        if raw_mol is None:
            print(
                f"[{folder_path.name}] RDKit error: MolFromXYZFile returned None for "
                f"{first_xyz_path.name}."
            )
            return None
        rdDetermineBonds.DetermineConnectivity(raw_mol)
        smiles_mol = Chem.MolFromSmiles(smiles)
        if smiles_mol is None:
            print(f"[{folder_path.name}] Error: unparseable SMILES {smiles!r}.")
            return None
        template_mol = AllChem.AssignBondOrdersFromTemplate(
            Chem.AddHs(smiles_mol), raw_mol,
        )
        Chem.SanitizeMol(template_mol)
    except Exception:  # noqa: BLE001
        print(f"[{folder_path.name}] Error building SMILES topology ({method}, {smiles!r}):")
        traceback.print_exc()
        return None

    template_mol.RemoveAllConformers()
    num_atoms = template_mol.GetNumAtoms()
    conf_to_orig_idx: dict[int, int] = {}
    for idx, path in indexed_files:
        try:
            coords = []
            with open(path, encoding="utf-8") as f:
                for line in f.readlines()[2:]:
                    parts = line.split()
                    if len(parts) == 4:
                        coords.append([float(x) for x in parts[1:]])
            if len(coords) != num_atoms:
                print(
                    f"[{folder_path.name}] Error: conformer {idx} has {len(coords)} atoms, "
                    f"but the topology has {num_atoms}."
                )
                return None
            conf = Chem.Conformer(num_atoms)
            for atom_i, pos in enumerate(coords):
                conf.SetAtomPosition(atom_i, Point3D(*pos))
            conf_to_orig_idx[template_mol.AddConformer(conf, assignId=True)] = idx
        except Exception:  # noqa: BLE001
            print(f"[{folder_path.name}] Error adding geometry for conformer {idx} ({path.name}):")
            traceback.print_exc()
            return None
    return template_mol, conf_to_orig_idx


def _rank_with_forcefield(
    folder_path: Path,
    template_mol: Chem.Mol,
    conf_to_orig_idx: dict[int, int],
    make_ff,
    method: str,
) -> list[int] | None:
    energies: list[tuple[int, float]] = []
    for conf_id, orig_idx in conf_to_orig_idx.items():
        try:
            ff = make_ff(template_mol, conf_id)
            if ff is None:
                print(
                    f"[{folder_path.name}] RDKit error: {method} force field is None "
                    f"for conformer {orig_idx}."
                )
                return None
            energies.append((orig_idx, ff.CalcEnergy()))
        except Exception:  # noqa: BLE001
            print(f"[{folder_path.name}] Exception computing {method} energy for conformer {orig_idx}:")
            traceback.print_exc()
            return None
    if len(energies) != 30:
        print(f"[{folder_path.name}] Error: computed {len(energies)} conformers, expected 30.")
        return None
    energies.sort(key=lambda x: x[1])
    return [item[0] for item in energies]


def _compute_uff_ranking(folder_path: Path) -> list[int] | None:
    """UFF energy ranking using SMILES topology + XYZ coordinates."""
    built = _build_template_with_conformers(folder_path, "UFF")
    if built is None:
        return None
    template_mol, conf_to_orig_idx = built
    return _rank_with_forcefield(
        folder_path, template_mol, conf_to_orig_idx,
        lambda mol, cid: AllChem.UFFGetMoleculeForceField(mol, confId=cid),
        "UFF",
    )


def get_mmff94_ranking(folder_path: Path) -> list[int] | None:
    """MMFF94 single-point ranking of conf indices (lowest first). Cached on disk."""
    smiles = _smiles_by_folder().get(folder_path.name)
    return _ranking_cached(
        folder_path, MMFF94_NAME, _compute_mmff94_ranking, smiles=smiles,
    )


def _compute_mmff94_ranking(folder_path: Path) -> list[int] | None:
    """MMFF94 energy ranking using SMILES topology + XYZ coordinates."""
    built = _build_template_with_conformers(folder_path, MMFF94_NAME)
    if built is None:
        return None
    template_mol, conf_to_orig_idx = built
    try:
        ff_props = AllChem.MMFFGetMoleculeProperties(template_mol)
    except Exception:  # noqa: BLE001
        print(f"[{folder_path.name}] Error: MMFFGetMoleculeProperties failed:")
        traceback.print_exc()
        return None
    if ff_props is None:
        print(f"[{folder_path.name}] RDKit error: MMFFGetMoleculeProperties returned None.")
        return None
    return _rank_with_forcefield(
        folder_path, template_mol, conf_to_orig_idx,
        lambda mol, cid: AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=cid),
        MMFF94_NAME,
    )


def get_gfnff_ranking(folder_path: Path) -> list[int] | None:
    """GFN-FF ranking of conf indices (lowest energy first). Cached on disk."""
    return _ranking_cached(folder_path, "GFN-FF", _compute_gfnff_ranking)


def _compute_gfnff_ranking(folder_path: Path) -> list[int] | None:
    if not folder_path.is_dir():
        return None
    if not shutil.which("xtb"):
        print(f"[GFN-FF] {folder_path.name}: 'xtb' not found on PATH; skipping molecule")
        return None
    xyz_files = list(folder_path.glob("*.xyz"))
    indexed_files = []
    for p in xyz_files:
        match = re.search(r'conf_(\d+)\.xyz$', p.name)
        if match:
            indexed_files.append((int(match.group(1)), p))
    indexed_files.sort(key=lambda x: x[0])
    if len(indexed_files) != 30:
        print(
            f"[GFN-FF] {folder_path.name}: found {len(indexed_files)}/30 conformers; "
            "skipping molecule"
        )
        return None
    energies = []
    for idx, path in indexed_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_xyz = Path(tmpdir) / "mol.xyz"
            try:
                shutil.copy(path, temp_xyz)
                res = subprocess.run(
                    ["xtb", "mol.xyz", "--gfnff"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                match_e = re.search(r"total energy\s+([-\d.]+)\s+Eh", res.stdout, re.IGNORECASE)
                if match_e:
                    energies.append((idx, float(match_e.group(1))))
                else:
                    print(
                        f"[GFN-FF] {path.name}: no 'total energy' line in xtb output; "
                        "skipping conformer"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[GFN-FF] {path.name}: xtb failed ({exc}); skipping conformer")
                continue
    if len(energies) != 30:
        print(
            f"[GFN-FF] {folder_path.name}: only {len(energies)}/30 energies from xtb; "
            "skipping molecule"
        )
        return None
    energies.sort(key=lambda x: x[1])
    return [item[0] for item in energies]


def _forcefield_ranking(folder_path: Path, method: str) -> list[int] | None:
    if method == "UFF":
        return get_uff_ranking(folder_path)
    if method == "GFN-FF":
        return get_gfnff_ranking(folder_path)
    return get_mmff94_ranking(folder_path)


def compute_forcefield_row(
    meta: dict,
    conformers_dir: Path,
    method: str,
    floors: list[float] | None = None,
) -> dict | None:
    answers = {}
    n_molecules = len(meta)

    if n_molecules == 0:
        print(f"[{method}] Error: metadata index is empty.")
        return None

    if not conformers_dir.is_dir():
        print(f"[{method}] Error: directory '{conformers_dir}' not found.")
        return None

    print(f"\n====== {method} ranking ======")
    print(f"[{method}] ranking cache: {conformers_dir / FF_CACHE_DIRNAME}")
    for mol_name, mol_meta in meta.items():
        folder_path = conformers_dir / mol_name

        if not folder_path.is_dir():
            continue

        print(f"-> Processing molecule: {mol_name}")
        if method in ("UFF", "GFN-FF"):
            cached = _load_cached_ranking(folder_path, method) is not None
            if cached:
                print("   (cache hit)")

        pred_rank = _forcefield_ranking(folder_path, method)

        if pred_rank is None or len(pred_rank) != 30:
            print(f"[{method}] ranking failed for {mol_name} (skipping).\n")
            continue

        pub_map = mol_meta.get("public_label_to_conf_index", {})
        conf_to_pub = {int(v): str(k) for k, v in pub_map.items()}
        pub_ranking = [conf_to_pub[idx] for idx in pred_rank if idx in conf_to_pub]

        if len(pub_ranking) == len(pred_rank):
            answers[mol_name] = pub_ranking
            print(f"-> Ranked {mol_name}.\n")
        else:
            print(f"-> Label mapping error for {mol_name}.\n")

    print(f"=== {method} summary ===")
    print(f"Ranked {len(answers)} of {n_molecules} molecules in metadata.\n")

    if len(answers) == 0:
        return None

    row = _score_tau(answers, meta)
    row = _with_reliable_floors(row, answers, meta, floors)
    row["_mol_metrics"] = _mol_metrics(answers, meta, floors)
    return row


def model_color(name: str) -> str:
    return color_for_display(name)


def model_marker(name: str) -> str:
    return marker_for_display(name)


def scatter_model_point(
    ax: plt.Axes,
    name: str,
    x: float,
    y: float,
    *,
    uniform_size: bool = False,
    marker: str | None = None,
    size: float | None = None,
    alpha: float = 1.0,
) -> None:
    color = model_color(name)
    used_marker = model_marker(name) if marker is None else marker
    used_size, edgewidth = scatter_style_for_display(name)
    if uniform_size:
        used_size, edgewidth = MARKER_SIZE, MARKER_EDGEWIDTH
    if size is not None:
        used_size = size
    ax.scatter(
        [x],
        [y],
        s=used_size,
        c=color,
        marker=used_marker,
        edgecolors="white",
        linewidths=edgewidth,
        alpha=alpha,
        zorder=5,
    )


def build_legend_handles(
    present_names: set[str],
    has_uff: bool = True,
    has_gfnff: bool = True,
    has_mmff: bool = False,
) -> list[Line2D]:
    handles: list[Line2D] = [
        Line2D([0], [0], color=REF_RANDOM, linestyle="--", linewidth=1.2, label="Random"),
    ]
    if has_uff:
        handles.append(Line2D([0], [0], color=REF_UFF, linestyle=":", linewidth=1.5, label="UFF"))
    if has_gfnff:
        handles.append(Line2D([0], [0], color=REF_GFNFF, linestyle="-.", linewidth=1.5, label="GFN-FF"))
    if has_mmff:
        handles.append(Line2D([0], [0], color=REF_MMFF, linestyle=REF_MMFF_LINESTYLE, linewidth=1.5, label="MMFF94"))

    for fam in VENDOR_FAMILIES:
        if not any(family_for_display(n) == fam for n in present_names):
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKER_DEFAULT,
                color="w",
                markerfacecolor=family_legend_color(fam, present_names),
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8,
                label=fam,
            )
        )
    if any(is_reasoning_or_thinking(n) for n in present_names):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKER_REASONING,
                color="w",
                markerfacecolor="#6b7280",
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8,
                label="Reasoning",
            )
        )
    return handles


def load_model_answers(
    filename: str,
    meta: dict,
    *,
    bundle_dir: Path,
    answers_dir: Path | None,
) -> dict:
    """Use the seed-42 bundle (API runs); ``answers_dir`` is an explicit fallback only."""
    bundle_path = bundle_dir / filename
    if bundle_path.is_file():
        answers = load_valid_answers(bundle_path, meta)
        if answers:
            return answers
    if answers_dir is not None:
        path = answers_dir / filename
        if path.is_file():
            return load_valid_answers(path, meta)
    return {}


def _load_seed_metas(
    seed_dirs: list[tuple[str, Path]],
    deltae_override: dict | None,
) -> dict[str, dict]:
    metas = {
        seed: augment_meta(
            load_metadata_index(bundle_dir / "prompt_metadata.jsonl"),
            deltae_override,
            strict=deltae_override is not None,
        )
        for seed, bundle_dir in seed_dirs
    }
    if not metas:
        return metas
    common = set.intersection(*(set(meta) for meta in metas.values()))
    return {
        seed: {molecule: meta[molecule] for molecule in common}
        for seed, meta in metas.items()
    }


def _model_specs(model_files: dict[str, str] | None) -> list[tuple[str, str | None]]:
    if model_files is not None:
        return [(name, filename) for name, filename in model_files.items()]
    return list(default_models_table())


def _score_tau(answers: dict, meta: dict) -> dict:
    taus: list[float] = []
    for mol, rnk in answers.items():
        tau, _ = kendall_for_molecule(meta[mol]["letter_to_deltaE_kcal_mol"], rnk)
        taus.append(float(tau))
    n = len(answers)
    return {
        "n": n,
        "n_molecules": len(meta),
        "tau": round(sum(taus) / n, 3) if n else float("nan"),
    }


def _random_tau_row(meta: dict) -> dict:
    n_mol = len(meta)
    return {
        "n": n_mol,
        "n_molecules": n_mol,
        "tau": 0.0,
        "partial": False,
    }


def _with_reliable_floors(
    row: dict,
    answers: dict,
    meta: dict,
    floors: list[float] | None,
) -> dict:
    if floors:
        for f in floors:
            row[f"tau_f{f}"] = round(mean_reliable_tau(answers, meta, f), 3)
    return row


def _tau_keys(floors: list[float] | None) -> list[str]:
    keys = ["tau"]
    if floors:
        keys.extend(f"tau_f{f}" for f in floors)
    return keys


def _mol_metrics(
    answers: dict,
    meta: dict,
    floors: list[float] | None,
) -> dict[str, dict[str, float]]:
    """Per-molecule τ (and reliable-pair τ at floors) for one seed."""
    floor_list = list(floors or [])
    out: dict[str, dict[str, float]] = {}
    for mol, rnk in answers.items():
        if mol not in meta:
            continue
        de = meta[mol]["letter_to_deltaE_kcal_mol"]
        rec: dict[str, float] = {}
        tau, _ = kendall_for_molecule(de, rnk)
        rec["tau"] = float(tau)
        if floor_list:
            scored = score_molecule(rnk, de, floor_list)
            if scored.get("valid"):
                for f in floor_list:
                    val = scored.get(f"tau_f{f}")
                    if val is not None and np.isfinite(val):
                        rec[f"tau_f{f}"] = float(val)
        out[mol] = rec
    return out


def _mol_means_across_seeds(
    per_seed_mol: list[dict[str, dict[str, float]]],
    key: str,
) -> list[float]:
    if not per_seed_mol:
        return []
    mols = set.intersection(*(set(seed_map) for seed_map in per_seed_mol))
    means: list[float] = []
    for mol in mols:
        vals = [
            seed_map[mol][key]
            for seed_map in per_seed_mol
            if mol in seed_map and key in seed_map[mol]
        ]
        if vals:
            means.append(float(sum(vals) / len(vals)))
    return means


def _mol_means_all_seeds(
    per_seed_mol: list[dict[str, dict[str, float]]],
    key: str,
) -> list[float]:
    """Per-molecule mean of ``key`` using only molecules present in EVERY seed."""
    if not per_seed_mol:
        return []
    mols = set.intersection(*(set(m) for m in per_seed_mol))
    out: list[float] = []
    for mol in sorted(mols):
        vals = [m[mol][key] for m in per_seed_mol if key in m[mol]]
        if len(vals) == len(per_seed_mol):
            out.append(float(sum(vals) / len(vals)))
    return out


def _attach_molecule_ci(
    row: dict,
    per_seed_mol: list[dict[str, dict[str, float]]],
    keys: list[str],
) -> dict:
    """95% t-interval half-width across molecules (seeds averaged first)."""
    for key in keys:
        mol_means = _mol_means_across_seeds(per_seed_mol, key)
        if len(mol_means) < 2:
            continue
        mean, _std, _se, lo, hi = mean_std_ci(np.asarray(mol_means, dtype=float))
        row[key] = mean
        row[f"{key}_ci"] = 0.5 * (hi - lo)
    return row


def _mean_seed_scores(
    per_seed: list[dict],
    *,
    name: str,
    n_molecules: int,
) -> dict:
    n_seeds = len(per_seed)
    row = {
        "name": name,
        "n": n_molecules,
        "n_molecules": n_molecules,
        "tau": round(sum(r["tau"] for r in per_seed) / n_seeds, 3),
        "partial": False,
    }
    extra = [k for k in per_seed[0] if k.startswith("tau_f")]
    for key in extra:
        row[key] = round(sum(r[key] for r in per_seed) / n_seeds, 3)
    if n_seeds > 1:
        for key in ["tau", *extra]:
            vals = [r[key] for r in per_seed if key in r]
            if len(vals) > 1:
                row[f"{key}_std"] = round(statistics.stdev(vals), 3)
    return row


def _forcefield_baseline(
    conformers_dir: Path,
    metas: dict[str, dict],
    method: str,
    floors: list[float] | None = None,
) -> dict | None:
    """Score a force field on each seed's label shuffle; mean if multiple seeds."""
    per_seed: list[dict] = []
    n_molecules = len(next(iter(metas.values())))
    for meta in metas.values():
        row = compute_forcefield_row(meta, conformers_dir, method, floors=floors)
        if row is None:
            return None
        per_seed.append(row)
    mol_seeds = [r.pop("_mol_metrics", {}) for r in per_seed]
    if len(per_seed) == 1:
        out = per_seed[0]
    else:
        out = _mean_seed_scores(per_seed, name=method, n_molecules=n_molecules)
    return _attach_molecule_ci(out, mol_seeds, _tau_keys(floors))


def _maybe_forcefields(
    conformers_dir: Path,
    metas: dict[str, dict],
    *,
    forcefields: str,
    floors: list[float] | None = None,
) -> tuple[dict | None, dict | None, dict | None]:
    if forcefields == "none":
        return None, None, None
    uff_row = (
        _forcefield_baseline(conformers_dir, metas, "UFF", floors=floors)
        if forcefields in ("uff", "both", "all")
        else None
    )
    gfnff_row = (
        _forcefield_baseline(conformers_dir, metas, "GFN-FF", floors=floors)
        if forcefields in ("gfnff", "both", "all")
        else None
    )
    mmff_row = (
        _forcefield_baseline(conformers_dir, metas, "MMFF94", floors=floors)
        if forcefields in ("mmff", "all")
        else None
    )
    return uff_row, gfnff_row, mmff_row


def _geom_only_meta(row: dict) -> dict:
    return {mol: val for mol, val in row.items() if mol not in PILOT_MOLECULE_IDS}


def collect_model_rows(
    bundle_dir: Path,
    conformers_dir: Path,
    answers_dir: Path | None = None,
    deltae_override: dict | None = None,
    *,
    three_seeds: bool = True,
    forcefields: str = "none",
    floors: list[float] | None = None,
    model_files: dict[str, str] | None = None,
) -> tuple[list[dict], dict, dict, dict, dict]:
    specs = _model_specs(model_files)
    if three_seeds:
        seed_dirs = geom_seed_dirs()
        metas = _load_seed_metas(seed_dirs, deltae_override)
        n_molecules = len(next(iter(metas.values())))

        rand_per_seed = [_random_tau_row(meta) for meta in metas.values()]
        rand = _mean_seed_scores(rand_per_seed, name="Random", n_molecules=n_molecules)

        rows: list[dict] = []
        n_meta = len(next(iter(metas.values())))
        for name, filename in specs:
            if filename is None:
                continue
            per_seed_mol: list[dict[str, dict[str, float]]] = []
            for seed, seed_bundle in seed_dirs:
                answers = load_valid_answers(seed_bundle / filename, metas[seed])
                per_seed_mol.append(_mol_metrics(answers, metas[seed], floors))

            row: dict = {"name": name, "n_molecules": n_meta, "partial": False}
            n_scored = 0
            for key in _tau_keys(floors):
                means = _mol_means_all_seeds(per_seed_mol, key)
                if not means:
                    continue
                mean, std, _se, lo, hi = mean_std_ci(np.asarray(means, dtype=float))
                row[key] = round(mean, 3)
                row[f"{key}_ci"] = 0.5 * (hi - lo)
                if len(means) > 1:
                    row[f"{key}_std"] = round(std, 3)
                if key.startswith("tau_f"):
                    n_scored = len(means)
            if n_scored == 0:
                n_scored = len(_mol_means_all_seeds(per_seed_mol, "tau"))
            row["n"] = n_scored
            row["pilot_partial"] = n_scored < n_meta
            rows.append(row)

        uff_row, gfnff_row, mmff_row = _maybe_forcefields(
            conformers_dir, metas, forcefields=forcefields, floors=floors,
        )
        return rows, rand, uff_row, gfnff_row, mmff_row

    meta = augment_meta(
        load_metadata_index(bundle_dir / "prompt_metadata.jsonl"),
        deltae_override,
        strict=deltae_override is not None,
    )
    n_molecules = len(meta)
    rand = _random_tau_row(meta)
    rand["name"] = "Random"

    rows: list[dict] = []
    for name, filename in specs:
        if filename is None:
            continue
        answers = load_model_answers(
            filename, meta, bundle_dir=bundle_dir, answers_dir=answers_dir,
        )
        row = _with_reliable_floors(_score_tau(answers, meta), answers, meta, floors)
        row["name"] = name
        row["partial"] = row["n"] < n_molecules
        rows.append(
            _attach_molecule_ci(
                row, [_mol_metrics(answers, meta, floors)], _tau_keys(floors),
            )
        )

    uff_row, gfnff_row, mmff_row = _maybe_forcefields(
        conformers_dir, {"42": meta}, forcefields=forcefields, floors=floors,
    )
    return rows, rand, uff_row, gfnff_row, mmff_row


COST_FLOOR = TAU_FLOOR
COST_TAU_KEY = TAU_KEY
MIN_MOLS_PER_SEED = 5


def _pred_path(seed: Path, answers_name: str) -> Path:
    tag = answers_name.removeprefix("answers_").removesuffix(".jsonl")
    return seed / f"predictions_{tag}.jsonl"


def _load_costs(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mol = row.get("molecule")
        cost = row.get("cost_total")
        if mol is None or cost is None:
            continue
        out[str(mol)] = float(cost)
    return out


def common_benchmark_molecules(
    seeds: list[Path] | None = None,
    *,
    deltae_override: dict | None = None,
) -> set[str]:
    seed_list = list(seeds or geom_seed_paths())
    metas = {
        seed: augment_meta(
            load_metadata_index(seed / "prompt_metadata.jsonl"),
            deltae_override,
            strict=deltae_override is not None,
        )
        for seed in seed_list
    }
    return set.intersection(*(set(meta) for meta in metas.values()))


def total_api_cost_by_display(
    *,
    group: str = "geom_benchmark",
    seeds: list[Path] | None = None,
    deltae_override: dict | None = None,
    min_mols_per_seed: int = MIN_MOLS_PER_SEED,
    require_reliable_tau: bool = True,
) -> tuple[dict[str, float], dict[str, list[str]], set[str]]:
    """Return display -> SUM cost over mol x seeds, skipped seeds, and cohort molecules.

    When require_reliable_tau is True (default), only molecules with a valid
    reliable-pair tau @ COST_FLOOR contribute. When False, every molecule in the
    27-molecule cohort (25 GEOM + mPM/mPP) is included.
    """
    seed_list = list(seeds or geom_seed_paths())
    metas: dict[Path, dict] = {
        seed: augment_meta(
            load_metadata_index(seed / "prompt_metadata.jsonl"),
            deltae_override,
            strict=deltae_override is not None,
        )
        for seed in seed_list
    }
    common = common_benchmark_molecules(seed_list, deltae_override=deltae_override)
    totals: dict[str, float] = {}
    skipped: dict[str, list[str]] = {}

    for spec in models_in_group(group):
        if not spec.answers:
            continue
        sum_cost = 0.0
        miss: list[str] = []
        for seed in seed_list:
            meta = metas[seed]
            answers = load_valid_answers(seed / spec.answers, meta)
            costs = _load_costs(_pred_path(seed, spec.answers))
            seed_sum = 0.0
            n = 0
            for mol in common:
                if mol not in answers or mol not in costs:
                    continue
                if require_reliable_tau:
                    scored = score_molecule(
                        answers[mol], meta[mol]["letter_to_deltaE_kcal_mol"], [COST_FLOOR]
                    )
                    if not scored.get("valid"):
                        continue
                    tau = scored.get(COST_TAU_KEY)
                    if tau is None:
                        continue
                seed_sum += costs[mol]
                n += 1
            if n < min_mols_per_seed:
                miss.append(seed.name)
                continue
            sum_cost += seed_sum
        if sum_cost > 0 and not miss:
            totals[spec.display] = sum_cost
        elif miss:
            skipped[spec.display] = miss
    return totals, skipped, common

