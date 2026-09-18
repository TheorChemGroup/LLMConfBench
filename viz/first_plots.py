#!/usr/bin/env python3

"""Generate the manuscript figures and tables (one command).

    python viz/first_plots.py

Produces only:
  figures/r2scan3c/molecules_27_3row.png (+ .svg)
  figures/r2scan3c/seed_spread_heatmap.png
  figures/r2scan3c/ConcretenessTau.png
  figures/r2scan3c/FullPlot.png
  figures/r2scan3c/energy_probe_mae_heatmap.png
  figures/r2scan3c/tanimoto_distance_distr.png
  figures/r2scan3c/ensemble_rmsd_tfd_vs_nrot.png
  figures/r2scan3c/deltae_energy_profiles.png
  figures/r2scan3c/deltae_pair_diff_profiles.png
  figures/r2scan3c/tau_floor2_table.png  (+ .tsv)
  figures/r2scan3c/ensemble_flexibility.tsv
  figures/r2scan3c/paired_llm_minus_uff_floor2.tsv
  figures/release_scatter/r2scan3c/tau_vs_release_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_params_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_cost_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_gpqa_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_scicode_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_arcagi1_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_arcagi2_square_w95_2.png
  figures/release_scatter/r2scan3c/tau_vs_benchmarks_facet_2.png
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HELPER_ROOT = PROJECT_ROOT / "viz_helper"


def _log_step_outputs(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, Path):
        print(f"  -> {result}", flush=True)
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            _log_step_outputs(item)


def _run_step(label: str, run: Callable[[], Any]) -> None:
    print(f"\n--- {label} ---", flush=True)
    _log_step_outputs(run())


def main() -> int:
    os.chdir(PROJECT_ROOT)
    sys.path[:0] = [str(PROJECT_ROOT), str(HELPER_ROOT)]

    import mol_study as ms
    from benchmark_corr_table import write_corr_tsv as write_benchmark_corr
    from benchmark_corr_table import write_facet_figure as write_benchmark_facet
    from concreteness import build_table as build_concreteness
    from concreteness import load_setup as load_concreteness_setup
    from energy_probe_mae_heatmap import plot_mae_heatmap as write_energy_probe
    from plot_dataset_diversity_panels import write_figures as write_diversity
    from plot_tau_vs_arcagi import collect_and_write as write_tau_arcagi
    from plot_tau_vs_benchmark import collect_and_write as write_tau_bench
    from plot_tau_vs_cost import collect_and_write as write_tau_cost
    from plot_tau_vs_release_square import collect_and_write as write_tau_square
    from seed_spread_heatmap import collect_and_write as write_seed_heatmap
    from tau_floor2_table import collect_and_write as write_floor2_table

    from viz.three_mol_rows import main as write_mol_grid

    def _concreteness_table():
        models, mol_to_label, mol_labels = load_concreteness_setup()
        return build_concreteness(models, mol_to_label, mol_labels)

    def _mol_study_figures():
        d, order, _mt = ms.load_prepared(ms.DEFAULT_DATA)
        return [
            ms.plot_combined(d, order, ms.DEFAULT_OUTDIR),
            ms.plot_full(d, order, ms.DEFAULT_OUTDIR),
        ]

    def _paired_vs_uff():
        subprocess.run(
            [sys.executable, "metrics/run_paired_vs_uff.py"],
            check=True,
        )

    steps: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("tau vs release (|ΔE| >= 2.0)", write_tau_square),
        ("tau vs params (|ΔE| >= 2.0)", lambda: write_tau_square(x_mode="params")),
        ("tau vs GPQA / SciCode", write_tau_bench),
        ("tau vs API cost", write_tau_cost),
        ("tau vs ARC-AGI-1 / ARC-AGI-2", lambda: write_tau_arcagi(benches=["1", "2"])),
        ("tau vs all benchmarks (facet)", write_benchmark_facet),
        ("tau benchmark correlations (R^2 table)", write_benchmark_corr),
        ("seed spread heatmap (all models, min@10, |ΔE| >= 2.0)", write_seed_heatmap),
        ("tau @ 2.0 table (mean / median / SD)", write_floor2_table),
        ("dataset diversity panels + ensemble_flexibility.tsv", write_diversity),
        ("energy probe MAE heatmap", write_energy_probe),
        ("concreteness table (explanation-level counts)", _concreteness_table),
        ("ConcretenessTau + FullPlot", _mol_study_figures),
        ("27-molecule grid (PNG+SVG)", write_mol_grid),
        ("paired vs UFF TSV", _paired_vs_uff),
    )
    for label, run in steps:
        _run_step(label, run)
    print(
        "\nDone. figures/release_scatter/r2scan3c/ and figures/r2scan3c/",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
