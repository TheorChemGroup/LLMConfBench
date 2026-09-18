#!/usr/bin/env python3

"""Square τ @ 2.0 vs ARC-AGI-1 / ARC-AGI-2 / ARC-AGI-3 (from arcagi.json).

Scores are percent strings (`~` stripped; ranges → midpoint), matched to
geom-benchmark display names via `models.yaml` shorts.

  python viz_helper/plot_tau_vs_arcagi.py
  python viz_helper/plot_tau_vs_arcagi.py --bench 1 2
  python viz_helper/plot_tau_vs_arcagi.py --arcagi arcagi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scores import parse_pct_score
from plot_metrics_vs_release import collect_model_rows
from plot_tau_vs_benchmark import FLOOR, Y_KEY, write_tau_vs_aa
from plot_tau_vs_release_square import floor_tag

from config.models_config import (
    geom_conformers_dir,
    geom_seed_dirs,
    models_by_id,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models

DEFAULT_ARCAGI = _REPO / "arcagi.json"
OUT_DIR = Path("figures/release_scatter/r2scan3c")


ARC_BENCHES: dict[str, tuple[str, str]] = {
    "1": ("ARC-AGI-1", "tau_vs_arcagi1_square_w95"),
    "2": ("ARC-AGI-2", "tau_vs_arcagi2_square_w95"),
    "3": ("ARC-AGI-3", "tau_vs_arcagi3_square_w95"),
}
JSON_KEY = {"1": "arc_agi_1", "2": "arc_agi_2", "3": "arc_agi_3"}


def load_arc_by_display(arcagi_path: Path) -> dict[str, dict[str, float]]:
    """display name → {arc_agi_1, arc_agi_2, arc_agi_3} fractions."""
    short_to_display = {m.short: m.display for m in models_by_id().values()}
    payload = json.loads(arcagi_path.read_text(encoding="utf-8"))
    arc_by_display: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for row in payload["benchmark_results"]:
        short = row["model"]
        display = short_to_display.get(short)
        if display is None:
            skipped.append(f"{short} (unknown short)")
            continue
        parsed = {
            k: parse_pct_score(row.get(k))
            for k in ("arc_agi_1", "arc_agi_2", "arc_agi_3")
        }
        arc_by_display[display] = {k: v for k, v in parsed.items() if v is not None}

    print(f"ARC-AGI models parsed: {len(arc_by_display)}")
    if skipped:
        print("skipped:", ", ".join(skipped))
    for d, scores in sorted(
        arc_by_display.items(), key=lambda kv: -kv[1].get("arc_agi_2", -1)
    ):
        bits = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"  {d:28s}  {bits}")
    return arc_by_display


def default_out(stem: str) -> Path:
    return OUT_DIR / f"{stem}_{floor_tag(FLOOR)}.png"


def collect_and_write(
    *,
    arcagi: Path = DEFAULT_ARCAGI,
    benches: list[str] | None = None,
    out: Path | None = None,
    forcefields: bool = True,
    baseline: str = "r2scan3c",
) -> list[Path]:
    chosen = list(benches) if benches else list(ARC_BENCHES)
    unknown = [b for b in chosen if b not in ARC_BENCHES]
    if unknown:
        raise SystemExit(f"Unknown bench {unknown}; choose from {list(ARC_BENCHES)}")
    if out is not None and len(chosen) != 1:
        raise SystemExit("--out is only valid with a single --bench")

    arc_by_display = load_arc_by_display(arcagi)
    deltae_override = load_deltae_override(resolve_deltae_json(baseline))
    models, rand, uff, gfnff, mmff = collect_model_rows(
        geom_seed_dirs()[0][1],
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for bench in chosen:
        xlabel, stem = ARC_BENCHES[bench]
        key = JSON_KEY[bench]
        scores_by_name = {
            name: vals[key]
            for name, vals in arc_by_display.items()
            if key in vals
        }
        n = sum(1 for m in models if m["name"] in scores_by_name and Y_KEY in m)
        print(f"\n=== {xlabel}: {n} models with τ + score ===")
        if n < 3:
            print(f"skip {xlabel}: need ≥3 points")
            continue
        dest = out if out is not None else default_out(stem)
        written.append(
            write_tau_vs_aa(
                dest,
                models,
                rand,
                uff,
                gfnff,
                mmff,
                scores_by_name,
                xlabel=xlabel,
                y_key=Y_KEY,
            )
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench",
        nargs="*",
        choices=list(ARC_BENCHES),
        default=None,
        help="ARC-AGI version(s) 1 2 3 (default: all)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG for a single --bench "
             "(default: figures/release_scatter/r2scan3c/tau_vs_arcagiN_square_w95_2.png)",
    )
    ap.add_argument(
        "--arcagi",
        type=Path,
        default=DEFAULT_ARCAGI,
        help=f"ARC-AGI scores JSON (default: {DEFAULT_ARCAGI})",
    )
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument(
        "--no-forcefields",
        action="store_false",
        dest="forcefields",
        help="Omit UFF / GFN-FF reference lines",
    )
    ap.set_defaults(forcefields=True)
    args = ap.parse_args()
    paths = collect_and_write(
        arcagi=args.arcagi,
        benches=args.bench,
        out=args.out,
        forcefields=args.forcefields,
        baseline=args.baseline,
    )
    if paths:
        print("\nSaved:")
        for p in paths:
            print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
