#!/usr/bin/env python3

"""Square τ @ 2.0 vs total API cost (release-scatter style, molecule 95% CI).

Total cost for the plot uses molecules with valid reliable-pair τ @ 2.0.
The script also logs full 27-molecule × 3-seed API cost from predictions_*.jsonl.

  python viz_helper/plot_tau_vs_cost.py
  python viz_helper/plot_tau_vs_cost.py --out figures/release_scatter/r2scan3c/tau_vs_cost_w95_2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_metrics_vs_release import collect_model_rows, total_api_cost_by_display
from plot_tau_vs_benchmark import FLOOR, write_tau_vs_aa
from plot_tau_vs_release_square import floor_tag

from config.models_config import (
    geom_conformers_dir,
    geom_seed_dirs,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models

DEFAULT_OUT = Path("figures/release_scatter/r2scan3c") / f"tau_vs_cost_square_w95_{floor_tag(FLOOR)}.png"


def log_cohort_api_costs(
    model_names: set[str],
    *,
    cohort_costs: dict[str, float],
    skipped: dict[str, list[str]],
    cohort_mols: set[str],
) -> None:
    """Print per-model API cost for the full cohort (25 GEOM + pilots × 3 seeds)."""
    n_seeds = len(list(geom_seed_dirs()))
    print(
        f"\n{len(cohort_mols)}-molecule cohort API cost "
        f"({len(cohort_mols)} mols × {n_seeds} seeds = {len(cohort_mols) * n_seeds} calls):"
    )
    print(f"{'model':28s} {'cost ($)':>10s}")
    ranked = sorted(
        model_names,
        key=lambda name: (-cohort_costs.get(name, -1.0), name),
    )
    for name in ranked:
        cost = cohort_costs.get(name)
        if cost is None:
            seeds = skipped.get(name)
            note = f"missing seeds {', '.join(seeds)}" if seeds else "no predictions"
            print(f"{name:28s} {'—':>10s}  ({note})")
            continue
        print(f"{name:28s} {cost:10.2f}")


def collect_and_write(
    out: Path | None = None,
    *,
    forcefields: bool = True,
    baseline: str = "r2scan3c",
) -> Path:
    bundle_dir = geom_seed_dirs()[0][1]
    deltae_override = load_deltae_override(resolve_deltae_json(baseline))
    models, rand, uff, gfnff, mmff = collect_model_rows(
        bundle_dir,
        geom_conformers_dir(),
        deltae_override=deltae_override,
        three_seeds=True,
        forcefields="all" if forcefields else "none",
        floors=[FLOOR],
        model_files=dict(
            complete_geom_models(
                deltae_override=deltae_override,
                warn_unregistered=False,
            )
        ),
    )
    cohort_costs, cohort_skipped, cohort_mols = total_api_cost_by_display(
        deltae_override=deltae_override,
        require_reliable_tau=False,
    )
    log_cohort_api_costs(
        {m["name"] for m in models},
        cohort_costs=cohort_costs,
        skipped=cohort_skipped,
        cohort_mols=cohort_mols,
    )
    cost_by_name, skipped, _ = total_api_cost_by_display(
        deltae_override=deltae_override,
    )
    if skipped:
        for name, seeds in sorted(skipped.items()):
            print(f"no full cost coverage for {name}: missing seeds {', '.join(seeds)}")
    missing = [m["name"] for m in models if m["name"] not in cost_by_name]
    if missing:
        print("no total API cost:", ", ".join(missing))

    dest = out or DEFAULT_OUT
    return write_tau_vs_aa(
        dest,
        models,
        rand,
        uff,
        gfnff,
        mmff,
        cost_by_name,
        xlabel="Total API cost ($)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help=f"default: {DEFAULT_OUT}")
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument(
        "--no-forcefields",
        action="store_false",
        dest="forcefields",
        help="Omit UFF / GFN-FF reference lines",
    )
    ap.set_defaults(forcefields=True)
    args = ap.parse_args()
    collect_and_write(args.out, forcefields=args.forcefields, baseline=args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
