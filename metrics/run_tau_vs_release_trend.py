#!/usr/bin/env python3

"""CLI: mean reliable-pair τ versus model release date (Spearman + R²).

Example:
  python metrics/run_tau_vs_release_trend.py
  python metrics/run_tau_vs_release_trend.py --floor 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import (
    family_for_display,
    geom_conformers_dir,
    geom_seed_paths,
    release_dates_by_display,
    resolve_deltae_json,
    short_label_for_display,
)
from metrics.kendall_tau_ranking import load_deltae_override
from metrics.tau_vs_release_trend import (
    leave_one_family_out,
    parse_release_date,
    summarize_tau_vs_date,
)
from viz.cohort import complete_geom_models
from viz_helper.plot_metrics_vs_release import collect_model_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floor", type=float, default=2.0)
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument("--leave-family-out", action="store_true")
    args = ap.parse_args()

    override = load_deltae_override(resolve_deltae_json(args.baseline))
    model_files = dict(
        complete_geom_models(deltae_override=override, warn_unregistered=False)
    )
    seed0 = geom_seed_paths()[0]
    rows, _rand, _uff, _gfn, _mmff = collect_model_rows(
        seed0,
        geom_conformers_dir(),
        deltae_override=override,
        three_seeds=True,
        forcefields="none",
        floors=[args.floor],
        model_files=model_files,
    )
    tau_key = f"tau_f{args.floor}"
    dates_map = release_dates_by_display()

    names: list[str] = []
    taus: list[float] = []
    dates: list = []
    families: list[str] = []
    for row in rows:
        name = row["name"]
        if name not in dates_map or dates_map[name] is None:
            print(f"skip {name}: no release_date", file=sys.stderr)
            continue
        if tau_key not in row:
            print(f"skip {name}: no {tau_key}", file=sys.stderr)
            continue
        names.append(name)
        taus.append(float(row[tau_key]))
        dates.append(dates_map[name])
        families.append(family_for_display(name))

    summary = summarize_tau_vs_date(taus, dates)
    print(f"n={summary.n} models  floor={args.floor}")
    print(
        f"Spearman ρ = {summary.spearman_rho:+.3f}  "
        f"p = {summary.spearman_p:.3g}"
    )
    print(
        f"Pearson  r = {summary.pearson_r:+.3f}  "
        f"p = {summary.pearson_p:.3g}  "
        f"R² = {summary.r_squared:.3f}"
    )
    print(f"OLS slope = {summary.slope_per_year:+.3f} τ per year")
    print()
    print(f"{'short':28s} {'family':12s} {'date':12s} {'τ':>7s}")
    order = sorted(range(len(names)), key=lambda i: parse_release_date(dates[i]))
    for i in order:
        d = parse_release_date(dates[i]).isoformat()
        print(
            f"{short_label_for_display(names[i]):28s} "
            f"{families[i]:12s} {d:12s} {taus[i]:+7.3f}"
        )

    if args.leave_family_out:
        print("\nLeave-one-family-out:")
        print(f"{'dropped':12s} {'n':>3s} {'ρ':>7s} {'R²':>7s} {'p_ρ':>9s}")
        for fam, s in leave_one_family_out(taus, dates, families).items():
            print(
                f"{fam:12s} {s.n:3d} {s.spearman_rho:+7.3f} "
                f"{s.r_squared:7.3f} {s.spearman_p:9.3g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
