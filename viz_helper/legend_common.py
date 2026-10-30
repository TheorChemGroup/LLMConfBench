#!/usr/bin/env python3
"""Common legend for the CRA panels (release, equal-HB, decoy, decoy-vs-equal-HB).

Those four scatters are drawn without per-panel legends so the legend can be
placed once for the group. This script writes the shared strip:

    python viz_helper/legend_common.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_tau_vs_release_square import _legend_handles

from config.models_config import resolve_deltae_json
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models
from viz.plot_style import FONTSIZE, ONE_HALF_COLUMN_WIDTH, SAVE_DPI, plt

OUT = _REPO / "figures" / "cra_common_legend.png"


def main() -> int:
    override = load_deltae_override(resolve_deltae_json("r2scan3c", None))
    names = {
        display
        for display, _ in complete_geom_models(
            deltae_override=override, warn_unregistered=False
        )
    }
    handles = _legend_handles(names)
    fig = plt.figure(figsize=(ONE_HALF_COLUMN_WIDTH, 1.0))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        fontsize=FONTSIZE,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=SAVE_DPI, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
