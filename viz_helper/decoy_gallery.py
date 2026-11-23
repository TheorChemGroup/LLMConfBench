"""Compose the Fig. 3a decoy gallery from pre-rendered conformer images.

The individual panels in ``figures/fig3_gallery`` are produced with xyzrender
by ``render_decoy_gallery.py``; this module assembles them into the published
layout.

    python viz_helper/decoy_gallery.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "figures" / "fig3_gallery"
OUT = REPO / "figures" / "fig3a_decoy_gallery.png"
sys.path[:0] = [str(REPO), str(REPO / "viz")]

from viz.plot_style import FONTSIZE, FONTSIZE_LABEL  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def energy_ranks() -> dict[int, pd.Series]:
    """Molecule number -> correct energy rank (1 = lowest) of every conformer."""
    d = pd.read_csv(REPO / "pair_runs_reliable.csv")
    u = d.drop_duplicates(["molecule_number", "conf1", "conf2"])
    out = {}
    for mol, g in u.groupby("molecule_number"):
        n = int(max(g.conf1.max(), g.conf2.max())) + 1
        a = np.zeros((len(g), n))
        for r, (i, j) in enumerate(zip(g.conf1, g.conf2)):
            a[r, i] = -1
            a[r, j] = 1
        e = np.linalg.lstsq(a, g.dE_signed.values, rcond=None)[0]
        out[int(mol)] = pd.Series(e).rank().astype(int)
    return out


def label_map(mol: int) -> dict[int, int]:
    """Public label -> conformer index, seed 42."""
    from pathlib import Path as _P
    sys.path[:0] = [str(REPO), str(REPO / "metrics")]
    from metrics.kendall_tau_ranking import (augment_meta, load_deltae_override,
                                             load_metadata_index)
    from config.models_config import resolve_deltae_json
    labels = pd.read_csv(REPO / "figures/seed_spread_heatmap_molecules_fp.tsv", sep="\t")
    mid = labels[labels.label == mol].molecule_id.iloc[0]
    ov = load_deltae_override(resolve_deltae_json("r2scan3c", None))
    meta = augment_meta(load_metadata_index(
        _P(REPO / "geom-qm9/prompt_30x30_NEW/prompt_metadata.jsonl")), ov, strict=True)
    return {int(k): int(v) for k, v in meta[mid]["public_label_to_conf_index"].items()}

# mol, colour, mechanism, dE, (lower label, H-bonded label)
# mechanism: MMFF94 terms with a positive (penalising) contribution above 2.0 kcal/mol,
# ordered by size
MOLS = [
    (4,  "#2e7d32", "penalized by torsion + VdW", 2.17, (13, 16)),
    (8,  "#1565c0", "penalized by torsion + angle + VdW", 2.10, (10, 13)),
    (16, "#c62828", "penalized by torsion",       2.57, (5, 13)),
    (21, "#ef6c00", "penalized by angle",         2.38, (29, 14)),
    (22, "#6a1b9a", "penalized by angle + VdW + torsion",   6.44, (14, 3)),
]


RANKS = energy_ranks()


def main() -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(9.4, 5.1))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.755, bottom=0.07,
                        wspace=0.03, hspace=0.30)

    for j, (mol, col, term, dE, (low_pub, hb_pub)) in enumerate(MOLS):
        for i, tag in enumerate(("hb", "low")):
            ax = axes[i, j]
            ax.imshow(mpimg.imread(ASSETS / f"mol{mol}_{tag}.png"))
            ax.set_axis_off()
        head = term.replace("penalized by ", "penalized by\n")
        axes[0, j].set_title(f"mol {mol} {head}", color=col, fontsize=9,
                             pad=4, linespacing=1.15)
        l2c = label_map(mol)
        rk = RANKS[mol]
        axes[0, j].text(0.5, -0.005, f"conf {hb_pub}",
                        transform=axes[0, j].transAxes, ha="center", va="top",
                        fontsize=8, color=col)
        axes[0, j].text(0.5, -0.088, f"correct rank {rk[l2c[hb_pub]]}",
                        transform=axes[0, j].transAxes, ha="center", va="top",
                        fontsize=8, color="black")
        axes[1, j].text(0.5, 1.02,
                        f"$\\Delta E_{{\\mathrm{{r^2SCAN\u20103c}}}}$ = +{dE:.2f} kcal/mol",
                        transform=axes[1, j].transAxes, ha="center", va="center",
                        fontsize=FONTSIZE - 1.5, color="black",
                        linespacing=1.25)
        axes[1, j].text(0.5, -0.005, f"conf {low_pub}",
                        transform=axes[1, j].transAxes, ha="center", va="top",
                        fontsize=8, color=col)
        axes[1, j].text(0.5, -0.088, f"correct rank {rk[l2c[low_pub]]}",
                        transform=axes[1, j].transAxes, ha="center", va="top",
                        fontsize=8, color="black")

    fig.text(0.020, 0.68, "higher energy", rotation=90, va="center",
             ha="center", fontsize=FONTSIZE_LABEL, color="black")
    fig.text(0.020, 0.23, "lower energy", rotation=90, va="center",
             ha="center", fontsize=FONTSIZE_LABEL, color="black")

    fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(f"Saved {main()}")
