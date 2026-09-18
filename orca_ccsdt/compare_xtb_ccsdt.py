#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from compare_baselines_subset import (
    label_to_conf_ranking,
    load_answers,
    load_deltae_json,
    load_metas,
    reliable_tau,
)

from config.models_config import (
    geom_conformers_dir,
    geom_seed_paths,
    model_files_by_display,
)

FLOORS = [0.5, 1.0, 1.5, 2.0]
SEED_DIRS = geom_seed_paths()
MODEL_FILES = model_files_by_display("geom_benchmark")
CONFORMERS_DIR = geom_conformers_dir()


def xtb_conf_energies(meta_row: dict) -> dict[int, float] | None:
    """Metadata xTB letter energies mapped to conformer indices."""
    de = meta_row.get("letter_to_deltaE_kcal_mol")
    pub = meta_row.get("public_label_to_conf_index")
    n = meta_row.get("n_conformers", 0)
    if not de or not pub or len(de) != n:
        return None
    out: dict[int, float] = {}
    try:
        for lab, ci in pub.items():
            out[int(ci)] = float(de[str(lab)])
    except (KeyError, ValueError, TypeError):
        return None
    return out if len(out) == n else None


def xtb_from_xyz(mol_dir: Path) -> dict[int, float] | None:
    """GEOM/xTB relative energies from XYZ comment lines (conf_index → kcal/mol)."""
    import re

    pat = re.compile(r"_conf_(\d+)\.xyz$")
    energy_pat = re.compile(r"Energy:\s*([-\d.eE+]+)")
    out: dict[int, float] = {}
    if not mol_dir.is_dir():
        return None
    for path in mol_dir.glob("*_conf_*.xyz"):
        m = pat.search(path.name)
        if not m:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            return None
        em = energy_pat.search(lines[1])
        if not em:
            return None
        out[int(m.group(1))] = float(em.group(1))
    return out if out else None


def load_xtb_energies(
    mols: list[str],
    metas0: dict,
    *,
    conformers_dir: Path = CONFORMERS_DIR,
) -> dict[str, dict[int, float]]:
    """Prefer seed metadata; fall back to XYZ Energy: for mols outside the LLM cohort."""
    xtb: dict[str, dict[int, float]] = {}
    for mol in mols:
        row = metas0.get(mol)
        e = xtb_conf_energies(row) if row is not None else None
        if e is None:
            e = xtb_from_xyz(conformers_dir / mol)
        if e is not None:
            xtb[mol] = e
    return xtb


def pair_agreement(ref: dict[int, float], query: dict[int, float], floor: float,
                   pair_basis: str) -> tuple[int, int]:
    """Count agree/disagree between `query` ordering and `ref` ordering.

    pair_basis: which energy map selects the pairs (its gaps must be >= floor).
    """
    basis = ref if pair_basis == "ref" else query
    agree = disc = 0
    confs = sorted(basis.keys())
    for a in range(len(confs)):
        for b in range(a + 1, len(confs)):
            ca, cb = confs[a], confs[b]
            if ca not in ref or cb not in ref or ca not in query or cb not in query:
                continue
            if abs(basis[ca] - basis[cb]) < floor:
                continue
            truth = (ref[ca] < ref[cb]) - (ref[ca] > ref[cb])
            pred = (query[ca] < query[cb]) - (query[ca] > query[cb])
            if truth == 0:
                continue
            if pred == truth:
                agree += 1
            else:
                disc += 1
    return agree, disc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r2scan-json", type=Path,
                    default=_REPO / "orca_r2scan-3c" / "deltaE_r2scan3c_kcal_mol.json")
    ap.add_argument("--ccsdt-json", type=Path,
                    default=_HERE / "deltaE_dlpno_ccsdt_kcal_mol.json")
    ap.add_argument("--out", type=Path, default=_HERE / "xtb_vs_r2scan_vs_ccsdt.txt")
    args = ap.parse_args()

    r2 = load_deltae_json(args.r2scan_json)
    cc = load_deltae_json(args.ccsdt_json)


    mols_calc = sorted(set(r2) & set(cc))
    metas0 = load_metas(SEED_DIRS[0])
    xtb = load_xtb_energies(mols_calc, metas0)
    missing_xtb = [m for m in mols_calc if m not in xtb]
    if missing_xtb:
        print(f"WARNING: no xTB energies for {missing_xtb}", file=sys.stderr)
    mols = [m for m in mols_calc if m in xtb]

    mols_llm = [m for m in mols if m in metas0]
    from_xyz = [m for m in mols if m not in metas0]
    print(
        f"{len(mols)} subset molecules with xTB + r2SCAN-3c + CCSD(T) "
        f"({len(mols_llm)} in LLM cohort"
        + (f"; xTB from XYZ for {', '.join(from_xyz)}" if from_xyz else "")
        + ")"
    )

    lines: list[str] = []
    lines.append(
        f"Baseline vs DLPNO-CCSD(T) gold standard on {len(mols)} molecules "
        f"(xTB from seed metadata, or XYZ Energy: for non-LLM mols)"
    )

    lines.append("")
    lines.append("A) Agreement with CCSD(T) on pairs selected by the CCSD(T) gaps (fair basis):")
    lines.append(f"{'floor':>12s} {'pairs':>7s} | {'r2SCAN-3c agree%':>16s} | {'xTB agree%':>10s}")
    for floor in FLOORS:
        n_pairs = ra = rd = xa = xd = 0
        for mol in mols:
            a1, d1 = pair_agreement(cc[mol], r2[mol], floor, "ref")
            a2, d2 = pair_agreement(cc[mol], xtb[mol], floor, "ref")
            n_pairs += a1 + d1
            ra += a1
            rd += d1
            xa += a2
            xd += d2
        rpct = 100.0 * ra / (ra + rd) if ra + rd else float("nan")
        xpct = 100.0 * xa / (xa + xd) if xa + xd else float("nan")
        lines.append(f"{floor:>12.1f} {n_pairs:>7d} | {rpct:>15.1f}% | {xpct:>9.1f}%")

    lines.append("")
    lines.append("B) Agreement with CCSD(T) on pairs selected by each baseline's OWN gaps:")
    lines.append(f"{'floor':>12s} {'pairs r2':>8s} {'r2 agree%':>10s} {'pairs xtb':>9s} {'xTB agree%':>10s}")
    for floor in FLOORS:
        ra = rd = xa = xd = 0
        for mol in mols:
            a1, d1 = pair_agreement(cc[mol], r2[mol], floor, "query")
            a2, d2 = pair_agreement(cc[mol], xtb[mol], floor, "query")
            ra += a1
            rd += d1
            xa += a2
            xd += d2
        rpct = 100.0 * ra / (ra + rd) if ra + rd else float("nan")
        xpct = 100.0 * xa / (xa + xd) if xa + xd else float("nan")
        lines.append(f"{floor:>12.1f} {ra + rd:>8d} {rpct:>9.1f}% {xa + xd:>9d} {xpct:>9.1f}%")

    lines.append("")
    lines.append("C) Kendall tau vs CCSD(T) conformer ranking (mean over molecules):")
    lines.append(
        f"{'baseline':>12s} {'full tau':>9s} {'tau@0.5':>8s} {'tau@1.0':>8s} "
        f"{'tau@1.5':>8s} {'tau@2.0':>8s}"
    )
    for name, emap in (("r2SCAN-3c", r2), ("xTB", xtb)):
        full, t05, t10, t15, t20 = [], [], [], [], []
        for mol in mols:
            base_rank = sorted(emap[mol], key=lambda c: emap[mol][c])
            if len(cc[mol]) != len(base_rank):
                continue
            full_tau = reliable_tau(cc[mol], base_rank, 0.0)
            if full_tau is not None:
                full.append(full_tau)
            for floor, acc in ((0.5, t05), (1.0, t10), (1.5, t15), (2.0, t20)):
                v = reliable_tau(cc[mol], base_rank, floor)
                if v is not None:
                    acc.append(v)
        lines.append(
            f"{name:>12s} {np.mean(full):>9.4f} {np.mean(t05):>8.4f} "
            f"{np.mean(t10):>8.4f} {np.mean(t15):>8.4f} {np.mean(t20):>8.4f}"
        )


    from scipy.stats import kendalltau as kt
    from scipy.stats import spearmanr

    metas = {s.name: load_metas(s) for s in SEED_DIRS}
    answers = {s.name: {m: load_answers(s / f) for m, f in MODEL_FILES.items()}
               for s in SEED_DIRS}
    lines.append("")
    lines.append(
        f"D) Model-ranking agreement (r2SCAN-3c vs CCSD(T) scores; "
        f"{len(mols_llm)} LLM-cohort mols × 3 seeds):"
    )
    lines.append(f"{'floor':>8s} {'Spearman rho':>13s} {'Kendall tau':>12s}")
    for floor in FLOORS:
        r2_scores: dict[str, float] = {}
        cc_scores: dict[str, float] = {}
        for model in MODEL_FILES:
            r2_taus: list[float] = []
            cc_taus: list[float] = []
            for s in SEED_DIRS:
                meta = metas[s.name]
                for mol in mols_llm:
                    row = meta.get(mol)
                    if row is None:
                        continue
                    cr = label_to_conf_ranking(row, answers[s.name][model].get(mol, []))
                    if cr is None:
                        continue
                    if mol in r2:
                        t = reliable_tau(r2[mol], cr, floor)
                        if t is not None:
                            r2_taus.append(t)
                    if mol in cc:
                        t = reliable_tau(cc[mol], cr, floor)
                        if t is not None:
                            cc_taus.append(t)
            r2_scores[model] = float(np.mean(r2_taus)) if r2_taus else float("nan")
            cc_scores[model] = float(np.mean(cc_taus)) if cc_taus else float("nan")
        vals_r = np.array([r2_scores[m] for m in MODEL_FILES], dtype=float)
        vals_c = np.array([cc_scores[m] for m in MODEL_FILES], dtype=float)
        rho, _ = spearmanr(vals_r, vals_c)
        tau, _ = kt(vals_r, vals_c)
        lines.append(f"{floor:>8.1f} {rho:>13.4f} {tau:>12.4f}")

    report = "\n".join(lines)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
