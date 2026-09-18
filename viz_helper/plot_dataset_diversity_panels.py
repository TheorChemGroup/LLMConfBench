#!/usr/bin/env python3

"""Dataset diversity figures for the GEOM-QM9 conformer benchmark (separate PNGs).

  A — Pairwise Tanimoto distance histogram (ECFP4 / Morgan r=2)
  B — Ensemble RMSD / TFD vs N_rot stratification
  C — 3D conformer overlays (rigid / moderate / flexible)
  D — r²SCAN-3c ΔE jitter profiles

  python viz_helper/plot_dataset_diversity_panels.py   # panels A–D + companion TSVs
  python viz/first_plots.py                            # all manuscript figures
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    AllChem,
    TorsionFingerprints,
    rdDetermineBonds,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Geometry import Point3D
from scipy.stats import gaussian_kde

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.models_config import (
    geom_conformers_dir,
    geom_seed_dirs,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import (
    load_deltae_override,
    load_metadata_index,
)
from viz.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    FONTSIZE,
    ONE_HALF_COLUMN_WIDTH,
    plt,
    style_ax,
)
from viz_helper.plot_metrics_vs_release import _smiles_by_folder

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
OUT_DIR = Path("figures/r2scan3c")
OUT_A = OUT_DIR / "tanimoto_distance_distr.png"
OUT_B = OUT_DIR / "ensemble_rmsd_tfd_vs_nrot.png"
OUT_D = OUT_DIR / "deltae_energy_profiles.png"
OUT_DD = OUT_DIR / "deltae_pair_diff_profiles.png"
OUT_ENSEMBLE_FLEX = OUT_DIR / "ensemble_flexibility.tsv"


PROBE_MOL_TABLE = OUT_DIR / "seed_spread_heatmap_molecules.tsv"

PANEL_C_MOLS = (
    ("OCC1_CCCCOC1", r"$N_{\mathrm{rot}}=1$ (rigid)"),
    ("CC__O__C_H__C_O__C__H__C_O", r"$N_{\mathrm{rot}}=3$ (moderate)"),
    ("C_C__H__CO_CCNC_O", r"$N_{\mathrm{rot}}=5$ (flexible)"),
)


PANEL_BD_FIGSIZE = (DOUBLE_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH * 0.95)
PANEL_BD_MARGINS = {"left": 0.08, "right": 0.90, "top": 0.97, "bottom": 0.10}
TSV_FLOAT_FMT = "%.3f"
JITTER_WIDTH = 0.28
DOT_COLOR = "#334155"
DOT_SIZE = 10
DOT_ALPHA = 0.70
RMSD_DOT_COLOR = "#0f766e"
RMSD_LABEL_COLOR = "#16a34a"
RMSD_SPINE_COLOR = "#000000"
RMSD_YMAX = 2.5
RMSD_YMIN = -0.05
TFD_DOT_COLOR = "#7c3aed"
TFD_LABEL_COLOR = "#7c3aed"
TFD_SPINE_COLOR = "#000000"
TFD_YMAX = 1.0
TFD_YMIN = -0.02
REF_CONF_ID = 0


def kb_t_kcal(temperature_k: float = 298.15) -> float:
    """Thermal energy k_B T in kcal/mol (gas constant R per mole)."""

    return 1.98720425864083e-3 * float(temperature_k)


KB_T_KCAL = kb_t_kcal(298.15)


def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def tanimoto_distance_matrix(fps) -> np.ndarray:
    """Pairwise Tanimoto distances d_ij = 1 − T_c(fp_i, fp_j); diagonal 0."""
    n = len(fps)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            dist[i, j] = dist[j, i] = d
    return dist


def condensed_pair_count(n: int) -> int:
    return n * (n - 1) // 2


def pairwise_mean_max(condensed) -> tuple[float, float]:
    """Mean and max over a condensed pairwise matrix (length n(n−1)/2)."""
    arr = np.asarray(condensed, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.max())


def ensemble_tfd_stats(mol: Chem.Mol) -> tuple[float, float]:
    """Mean/max pairwise Torsion Fingerprint Deviation over conformers."""
    n = mol.GetNumConformers()
    mat = TorsionFingerprints.GetTFDMatrix(mol)
    if len(mat) != condensed_pair_count(n):
        raise ValueError(
            f"TFD matrix length {len(mat)} != C({n},2)={condensed_pair_count(n)}"
        )
    return pairwise_mean_max(mat)


def ensemble_heavy_rmsd_stats(mol: Chem.Mol) -> tuple[float, float]:
    """Mean/max pairwise heavy-atom RMSD (Å) over conformers."""
    n = mol.GetNumConformers()
    heavy = heavy_atom_indices(mol)
    if not heavy:
        raise ValueError("molecule has no heavy atoms")
    mat = AllChem.GetConformerRMSMatrix(mol, atomIds=heavy, prealigned=False)
    if len(mat) != condensed_pair_count(n):
        raise ValueError(
            f"RMSD matrix length {len(mat)} != C({n},2)={condensed_pair_count(n)}"
        )
    return pairwise_mean_max(mat)


def conf_folder_for(mol_name: str) -> Path | None:
    folder = geom_conformers_dir() / mol_name
    return folder if folder.is_dir() else None


def load_ensemble_mol(folder: Path, smiles: str) -> Chem.Mol:
    indexed: list[tuple[int, Path]] = []
    for path in folder.glob("*.xyz"):
        match = re.search(r"conf_(\d+)\.xyz$", path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda x: x[0])
    if not indexed:
        raise FileNotFoundError(f"no conf_*.xyz in {folder}")

    raw = Chem.MolFromXYZFile(str(indexed[0][1]))
    if raw is None:
        raise ValueError(f"MolFromXYZFile failed: {indexed[0][1].name}")
    rdDetermineBonds.DetermineConnectivity(raw)
    smiles_mol = Chem.MolFromSmiles(smiles)
    if smiles_mol is None:
        raise ValueError(f"bad SMILES {smiles!r}")
    template = AllChem.AssignBondOrdersFromTemplate(Chem.AddHs(smiles_mol), raw)
    Chem.SanitizeMol(template)
    template.RemoveAllConformers()
    n_atoms = template.GetNumAtoms()

    for _idx, path in indexed:
        coords: list[list[float]] = []
        for line in path.read_text(encoding="utf-8").splitlines()[2:]:
            parts = line.split()
            if len(parts) == 4:
                coords.append([float(x) for x in parts[1:]])
        if len(coords) != n_atoms:
            raise ValueError(f"{path.name}: {len(coords)} atoms, topology has {n_atoms}")
        conf = Chem.Conformer(n_atoms)
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(*pos))
        template.AddConformer(conf, assignId=True)
    return template


def collect_metrics() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return ensemble table, pairwise Tanimoto-distance matrix, molecule names."""
    meta = load_metadata_index(geom_seed_dirs()[0][1] / "prompt_metadata.jsonl")
    molecules = sorted(set(meta))
    smiles_map = _smiles_by_folder()

    rows: list[dict] = []
    fps = []
    names: list[str] = []
    for mol_name in molecules:
        smiles = smiles_map.get(mol_name)
        folder = conf_folder_for(mol_name)
        if smiles is None or folder is None:
            continue
        bare = Chem.MolFromSmiles(smiles)
        if bare is None:
            continue
        ensemble = load_ensemble_mol(folder, smiles)
        tfd_mean, tfd_max = ensemble_tfd_stats(ensemble)
        rmsd_mean, rmsd_max = ensemble_heavy_rmsd_stats(ensemble)
        rows.append(
            {
                "molecule": mol_name,
                "smiles": smiles,
                "n_confs": ensemble.GetNumConformers(),
                "NumRotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(bare)),
                "TFD_mean": tfd_mean,
                "TFD_max": tfd_max,
                "RMSD_mean": rmsd_mean,
                "RMSD_max": rmsd_max,
            }
        )
        fps.append(MORGAN.GetFingerprint(bare))
        names.append(mol_name)

    dist = tanimoto_distance_matrix(fps)
    return pd.DataFrame(rows), dist, names


def molecule_panel_order(ens: pd.DataFrame) -> list[str]:
    """Shared x-order for RMSD/TFD and ΔE panels (by N_rot, then RMSD_mean)."""
    ordered = ens.sort_values(
        ["NumRotatableBonds", "RMSD_mean"], ascending=[True, True]
    )
    return list(ordered["molecule"])


def probe_molecule_order(table: Path = PROBE_MOL_TABLE) -> list[str]:
    """Molecule ids in the seed-spread / energy-probe heatmap row order."""
    if not table.is_file():
        return []
    out: list[str] = []
    for line in table.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append(parts[1])
    return out


def _ordered_panel_context(
    ens: pd.DataFrame,
    dist: np.ndarray,
    names: list[str],
) -> tuple[list[str], dict[str, str], dict[str, int], np.ndarray, list[str]]:
    smiles_map = {r["molecule"]: r["smiles"] for _, r in ens.iterrows()}
    order = molecule_panel_order(ens)
    id_by_name = {mol: i for i, mol in enumerate(order, start=1)}
    name_index = {n: i for i, n in enumerate(names)}
    order_idx = [name_index[m] for m in order]
    dist_ordered = dist[np.ix_(order_idx, order_idx)]
    return order, smiles_map, id_by_name, dist_ordered, list(order)


def write_ensemble_flexibility_tsv(
    order: list[str],
    ens: pd.DataFrame,
    smiles_map: dict[str, str],
    out: Path = OUT_ENSEMBLE_FLEX,
) -> Path:
    """Flexibility metrics in panel x-order (same ids as ensemble_rmsd_tfd_vs_nrot)."""
    by_smiles = ens.drop(columns=["molecule"]).set_index("smiles")
    rows: list[dict] = []
    for i, mol in enumerate(order, start=1):
        smiles = smiles_map[mol]
        metrics = by_smiles.loc[smiles]
        rows.append(
            {
                "id": i,
                "smiles": smiles,
                "n_confs": int(metrics["n_confs"]),
                "NumRotatableBonds": int(metrics["NumRotatableBonds"]),
                "TFD_mean": float(metrics["TFD_mean"]),
                "TFD_max": float(metrics["TFD_max"]),
                "RMSD_mean": float(metrics["RMSD_mean"]),
                "RMSD_max": float(metrics["RMSD_max"]),
            }
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format=TSV_FLOAT_FMT)
    print(f"Saved {out}")
    return out


def _apply_id_xticks(ax, n: int) -> None:
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i) for i in range(1, n + 1)], fontsize=7)
    ax.tick_params(axis="x", length=3)


def per_conformer_rmsd_to_ref(mol: Chem.Mol, ref_id: int = REF_CONF_ID) -> np.ndarray:
    heavy = heavy_atom_indices(mol)
    n = mol.GetNumConformers()
    out = np.zeros(n, dtype=float)
    for cid in range(n):
        if cid == ref_id:
            continue
        out[cid] = float(
            AllChem.GetConformerRMS(mol, ref_id, cid, atomIds=heavy, prealigned=False)
        )
    return out


def per_conformer_tfd_to_ref(mol: Chem.Mol, ref_id: int = REF_CONF_ID) -> np.ndarray:
    n = mol.GetNumConformers()
    out = np.zeros(n, dtype=float)
    others = [cid for cid in range(n) if cid != ref_id]
    if not others:
        return out
    vals = TorsionFingerprints.GetTFDBetweenConformers(mol, [ref_id], others)
    for cid, val in zip(others, vals):
        out[cid] = float(val)
    return out


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=2000, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def write_tanimoto_matrix(
    dist: np.ndarray,
    names: list[str],
    out: Path = OUT_A,
) -> Path:
    tri_i, tri_j = np.triu_indices(len(names), k=1)
    vals = dist[tri_i, tri_j]

    fig, ax = plt.subplots(figsize=(ONE_HALF_COLUMN_WIDTH * 1.1, ONE_HALF_COLUMN_WIDTH * 0.85))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)
    ax.hist(vals, bins=18, density=False, color="#fdba74", edgecolor="white", alpha=0.85)
    try:
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), color="#9a3412", lw=1.2)
    except Exception:  # noqa: BLE001, S110
        pass
    mean_d = float(vals.mean())
    median_d = float(np.median(vals))
    print(
        f"Tanimoto distance (1-T_c): n={len(vals)}  "
        f"mean={mean_d:.3f}  median={median_d:.3f}  "
        f"min={float(vals.min()):.3f}  max={float(vals.max()):.3f}"
    )
    ax.set_xlim(0.55, 1.02)
    ax.set_xlabel(r"Tanimoto distance $(1-T_c)$", fontsize=FONTSIZE)
    ax.set_ylabel("Pairwise count", fontsize=FONTSIZE)
    style_ax(ax, square=False)
    return _save(fig, out)


def write_rmsd_tfd(
    order: list[str],
    smiles_map: dict[str, str],
    out: Path = OUT_B,
) -> Path:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=PANEL_BD_FIGSIZE)
    fig.subplots_adjust(**PANEL_BD_MARGINS)
    ax2 = ax.twinx()

    for i, mol_name in enumerate(order):
        folder = conf_folder_for(mol_name)
        if folder is None:
            continue
        mol = load_ensemble_mol(folder, smiles_map[mol_name])
        rms = per_conformer_rmsd_to_ref(mol)
        tfd = per_conformer_tfd_to_ref(mol)
        j_rms = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(rms))
        j_tfd = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(tfd))
        ax.scatter(
            np.full(len(rms), i) + j_rms, rms,
            s=DOT_SIZE, c=RMSD_DOT_COLOR, alpha=DOT_ALPHA, linewidths=0, zorder=3,
        )
        ax2.scatter(
            np.full(len(tfd), i) + j_tfd, tfd,
            s=DOT_SIZE, c=TFD_DOT_COLOR, alpha=DOT_ALPHA, linewidths=0, zorder=3,
        )

    _apply_id_xticks(ax, len(order))
    ax.set_xlabel(
        r"Molecule id (seed-spread / energy-probe heatmap order)",
        fontsize=FONTSIZE,
    )
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Heavy-atom RMSD (Å)", fontsize=FONTSIZE, color=RMSD_LABEL_COLOR)
    ax2.set_ylabel("Torsion fingerprint deviation (TFD)", fontsize=FONTSIZE, color=TFD_LABEL_COLOR)
    ax.set_ylim(RMSD_YMIN, RMSD_YMAX)
    ax2.set_ylim(TFD_YMIN, TFD_YMAX)
    ax.set_yticks(np.arange(0.0, RMSD_YMAX + 0.01, 0.5))
    ax2.set_yticks(np.arange(0.0, TFD_YMAX + 0.01, 0.2))
    style_ax(ax, square=False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["left"].set_color(RMSD_SPINE_COLOR)
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        colors=RMSD_SPINE_COLOR,
        labelcolor=RMSD_LABEL_COLOR,
        direction="out",
        length=3,
        width=0.6,
        labelsize=FONTSIZE,
    )
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.9)
    ax2.spines["right"].set_color(TFD_SPINE_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(
        axis="y",
        which="both",
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        colors=TFD_SPINE_COLOR,
        labelcolor=TFD_LABEL_COLOR,
        direction="out",
        length=3,
        width=0.6,
        labelsize=FONTSIZE,
    )
    return _save(fig, out)


def _draw_3d_ensemble(ax, mol: Chem.Mol, color: str) -> None:
    AllChem.AlignMolConformers(mol)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    for cid in range(mol.GetNumConformers()):
        conf = mol.GetConformer(cid)
        pts = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ]
        )
        alpha = 0.08 if cid else 0.55
        lw = 0.4 if cid else 1.0
        for a, b in bonds:
            seg = pts[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, lw=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:  # noqa: BLE001, S110
        pass
    ax.view_init(elev=18, azim=35)


def write_deltae_profiles(
    order: list[str],
    deltae: dict,
    out: Path = OUT_D,
) -> Path:
    fig, ax = plt.subplots(figsize=PANEL_BD_FIGSIZE)
    fig.subplots_adjust(**PANEL_BD_MARGINS)
    rng = np.random.default_rng(0)
    for i, mol_name in enumerate(order):
        energies = deltae.get(mol_name)
        if not energies:
            continue
        vals = np.asarray([float(v) for v in energies.values()], dtype=float)
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA, linewidths=0, zorder=3,
        )
    ax.axhline(KB_T_KCAL, color="#ef4444", ls="--", lw=1.1, zorder=2)
    _apply_id_xticks(ax, len(order))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(r"$\Delta E$ (kcal/mol), r$^2$SCAN-3c", fontsize=1.5 * FONTSIZE)
    ax.set_ylim(-0.15, max(6.0, ax.get_ylim()[1]))
    style_ax(ax, square=False)
    return _save(fig, out)


def write_deltae_pair_diff_profiles(
    order: list[str],
    deltae: dict,
    out: Path = OUT_DD,
    *,
    floor: float = 2.0,
) -> Path:
    """Pairwise |ΔΔE| dots per molecule with the reliability floor (2.0 kcal/mol)."""
    fig, ax = plt.subplots(figsize=PANEL_BD_FIGSIZE)
    fig.subplots_adjust(**PANEL_BD_MARGINS)
    rng = np.random.default_rng(0)
    all_dd: list[float] = []
    for i, mol_name in enumerate(order):
        energies = deltae.get(mol_name)
        if not energies:
            continue
        vals = np.asarray([float(v) for v in energies.values()], dtype=float)
        iu, ju = np.triu_indices(len(vals), k=1)
        dd = np.abs(vals[iu] - vals[ju])
        if dd.size == 0:
            continue
        all_dd.extend(dd.tolist())
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=dd.size)
        ax.scatter(
            np.full(dd.size, i) + jitter, dd,
            s=DOT_SIZE, c=DOT_COLOR, alpha=0.22, linewidths=0, zorder=3,
        )
    ax.axhline(floor, color="#ef4444", ls="--", lw=1.1, zorder=2)
    _apply_id_xticks(ax, len(order))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(r"pairwise $|\Delta\Delta E|$ (kcal/mol), r$^2$SCAN-3c", fontsize=1.5  * FONTSIZE)
    top = max(6.0, float(np.percentile(all_dd, 99.5)) if all_dd else 6.0)
    ax.set_ylim(-0.15, top * 1.05)
    style_ax(ax, square=False)
    return _save(fig, out)


def write_figures(out_dir: Path = OUT_DIR) -> list[Path]:
    ens, dist, names = collect_metrics()
    order, smiles_map, _id_by_name, dist_ordered, names_ordered = _ordered_panel_context(
        ens, dist, names,
    )
    deltae = load_deltae_override(resolve_deltae_json("r2scan3c")) or {}
    out_dir = Path(out_dir)
    probe_order = [m for m in probe_molecule_order() if m in smiles_map] or order

    return [
        write_ensemble_flexibility_tsv(probe_order, ens, smiles_map, out_dir / OUT_ENSEMBLE_FLEX.name),
        write_tanimoto_matrix(dist_ordered, names_ordered, out_dir / OUT_A.name),
        write_rmsd_tfd(probe_order, smiles_map, out_dir / OUT_B.name),
        write_deltae_profiles(probe_order, deltae, out_dir / OUT_D.name),
        write_deltae_pair_diff_profiles(probe_order, deltae, out_dir / OUT_DD.name),
    ]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    write_figures(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
