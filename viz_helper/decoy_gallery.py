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

from viz.plot_style import FONTSIZE, FONTSIZE_LABEL, TEXT_COLOR  # noqa: E402

# mol, colour, mechanism, dE, H-bond geometry, (lower label, H-bonded label)
MOLS = [
    (4,  "#2e7d32", "penalized by torsion + VdW", 2.17, "2.00 Å / 122°", (13, 16)),
    (8,  "#1565c0", "penalized by torsion + VdW", 2.10, "2.04 Å / 146°", (10, 13)),
    (16, "#c62828", "penalized by torsion",       2.57, "1.87 Å / 162°", (5, 13)),
    (21, "#ef6c00", "penalized by angle",         2.38, "2.00 Å / 164°", (29, 14)),
    (22, "#6a1b9a", "penalized by angle + VdW",   6.44, "1.87 Å / 157°", (14, 3)),
]


def main() -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(9.4, 5.1))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.80, bottom=0.07,
                        wspace=0.03, hspace=0.30)

    for j, (mol, col, term, dE, geom, (low_pub, hb_pub)) in enumerate(MOLS):
        for i, tag in enumerate(("hb", "low")):
            ax = axes[i, j]
            ax.imshow(mpimg.imread(ASSETS / f"mol{mol}_{tag}.png"))
            ax.set_axis_off()
        axes[0, j].set_title(f"mol {mol}\n{term}", color=col,
                             fontsize=9.5, pad=5, linespacing=1.2)
        axes[0, j].text(0.5, -0.005, f"conf {hb_pub}",
                        transform=axes[0, j].transAxes,
                        ha="center", va="top", fontsize=8, color="#555555")
        axes[0, j].text(0.5, -0.105, f"ΔE = +{dE:.2f} kcal/mol\n{geom}",
                        transform=axes[0, j].transAxes, ha="center", va="top",
                        fontsize=FONTSIZE - 1.5, color=TEXT_COLOR,
                        linespacing=1.25)
        axes[1, j].text(0.5, -0.005, f"conf {low_pub}",
                        transform=axes[1, j].transAxes,
                        ha="center", va="top", fontsize=8, color="#555555")

    fig.text(0.020, 0.68, "higher energy", rotation=90, va="center",
             ha="center", fontsize=FONTSIZE_LABEL, color=TEXT_COLOR)
    fig.text(0.020, 0.23, "lower energy", rotation=90, va="center",
             ha="center", fontsize=FONTSIZE_LABEL, color=TEXT_COLOR)

    fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(f"Saved {main()}")
