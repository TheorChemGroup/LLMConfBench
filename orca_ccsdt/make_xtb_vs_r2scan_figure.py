from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))
from compare_baselines_subset import load_deltae_json, reliable_tau
from compare_xtb_ccsdt import SEED_DIRS, load_metas, load_xtb_energies

from viz.plot_style import (
    STEM_COLOR,
    legend_below,
    plt,
    style_ax,
)

FLOORS = [0.5, 1.0, 1.5, 2.0]
COL_R2 = "#0f2d7a"
COL_XTB = "#eab308"


def main() -> None:
    r2 = load_deltae_json(ROOT / "orca_r2scan-3c/deltaE_r2scan3c_kcal_mol.json")
    cc = load_deltae_json(_HERE / "deltaE_dlpno_ccsdt_kcal_mol.json")
    metas0 = load_metas(SEED_DIRS[0])
    mols_calc = sorted(set(r2) & set(cc))
    xtb = load_xtb_energies(mols_calc, metas0)
    mols = sorted(set(mols_calc) & set(xtb))
    print(f"{len(mols)} molecules for SI figure (xTB+r2SCAN+CCSD(T))")

    r_agree, x_agree = [], []
    for floor in FLOORS:
        ra = rd = xa = xd = 0
        for mol in mols:
            confs = sorted(cc[mol].keys())
            for a in range(len(confs)):
                for b in range(a + 1, len(confs)):
                    ca, cb = confs[a], confs[b]
                    if ca not in r2[mol] or cb not in r2[mol] or ca not in xtb[mol] or cb not in xtb[mol]:
                        continue
                    if abs(cc[mol][ca] - cc[mol][cb]) < floor:
                        continue
                    t = (cc[mol][ca] < cc[mol][cb]) - (cc[mol][ca] > cc[mol][cb])
                    if t == 0:
                        continue
                    pr = (r2[mol][ca] < r2[mol][cb]) - (r2[mol][ca] > r2[mol][cb])
                    px = (xtb[mol][ca] < xtb[mol][cb]) - (xtb[mol][ca] > xtb[mol][cb])
                    ra += pr == t
                    rd += pr != t
                    xa += px == t
                    xd += px != t
        r_agree.append(100.0 * ra / (ra + rd))
        x_agree.append(100.0 * xa / (xa + xd))

    r_tau, x_tau = [], []
    for floor in FLOORS:
        rv, xv = [], []
        for mol in mols:
            base_rank = sorted(r2[mol], key=lambda c: r2[mol][c])
            t = reliable_tau(cc[mol], base_rank, floor)
            if t is not None:
                rv.append(t)
            base_rank = sorted(xtb[mol], key=lambda c: xtb[mol][c])
            t = reliable_tau(cc[mol], base_rank, floor)
            if t is not None:
                xv.append(t)
        r_tau.append(float(sum(rv)) / len(rv))
        x_tau.append(float(sum(xv)) / len(xv))

    floors = np.asarray(FLOORS, dtype=float)
    off = 0.08

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    ax.vlines(floors - off, 60.0, r_agree, color=STEM_COLOR, linewidth=1.1, zorder=1)
    ax.vlines(floors + off, 60.0, x_agree, color=STEM_COLOR, linewidth=1.1, zorder=1)
    ax.scatter(
        floors - off, r_agree, s=38, c=COL_R2, zorder=3,
        edgecolors="white", linewidths=0.35, label="r2SCAN-3c",
    )
    ax.scatter(
        floors + off, x_agree, s=38, c=COL_XTB, marker="s", zorder=3,
        edgecolors="white", linewidths=0.35, label="xTB",
    )
    ax.set_ylim(60, 101)
    ax.set_xticks(FLOORS)
    ax.set_xlabel("Reliability floor |ΔE| (kcal/mol)")
    ax.set_ylabel("Agreement with DLPNO-CCSD(T), %")
    style_ax(ax)

    ax = axes[1]
    ax.plot(floors, r_tau, "o-", label="r2SCAN-3c", color=COL_R2, linewidth=1.4, markersize=6)
    ax.plot(floors, x_tau, "s--", label="xTB", color=COL_XTB, linewidth=1.4, markersize=6)
    ax.set_xticks(FLOORS)
    ax.set_xlabel("Reliability floor |ΔE| (kcal/mol)")
    ax.set_ylabel("Conformers Ranking Accuracy vs CCSD(T)")
    ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=0)
    style_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_below(fig, ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    out = _HERE / "xtb_vs_r2scan_vs_ccsdt.png"
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {out}")
    print("r2SCAN-3c agree%:", [f"{v:.1f}" for v in r_agree])
    print("xTB agree%:      ", [f"{v:.1f}" for v in x_agree])
    print("r2SCAN-3c tau:   ", [f"{v:.4f}" for v in r_tau])
    print("xTB tau:         ", [f"{v:.4f}" for v in x_tau])


if __name__ == "__main__":
    main()
