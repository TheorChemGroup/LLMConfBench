#!/usr/bin/env python3

"""Generate the manuscript figures and tables (one command).

    python viz/first_plots.py

Produces:
  figures/molecules_27_3row.png (+ .svg)
  figures/cra_fair_vs_release.png       (equal-HB metric)
  figures/cra_allpairs_vs_release.png   (all-pairs metric)
  figures/cra_vs_params.png             (equal-HB metric)
  figures/cra_vs_arcagi1.png            (equal-HB metric)
  figures/cra_vs_arcagi2.png            (equal-HB metric)
  figures/cra_vs_cost.png               (equal-HB metric)
  figures/cra_vs_gpqa.png               (equal-HB metric)
  figures/cra_vs_scicode.png            (equal-HB metric)
  figures/cra_vs_benchmarks_facet.png   (equal-HB metric)
  figures/cra_common_legend.png
  figures/cra_unconventional_vs_release.png
  figures/cra_conventional_vs_release.png
  figures/seed_spread_heatmap.png
  figures/tanimoto_distance_distr.png
  figures/ensemble_rmsd_tfd_vs_nrot.png
  figures/deltae_energy_profiles.png
  figures/deltae_pair_diff_profiles.png
  figures/energy_probe_mae_heatmap.png
  figures/ConcretenessCRA_GPT5_square.png
  figures/ConcretenessCRA_GPT6_square.png
  figures/ConcretenessCRA_all_horizontal.png
  figures/FullPlot.png
  figures/FullPlotConcepts.png
  figures/ensemble_flexibility.tsv
  figures/cra_floor2_table.tsv
  figures/paired_llm_minus_uff_floor2.tsv
  figures/terms_old_neutral_museglimmer30b.tsv (+ _agg.tsv)
  figures/models_table.tsv
  figures/models_reasoning_params.tsv
  figures/*_fp.tsv                        (fair-pair variants)

Intermediate tables written by the same steps: concreteness_vs_cra.tsv,
seed_spread_heatmap_molecules.tsv, cra_benchmark_correlations_2.tsv,
concept_counts_vs_cra.tsv.
"""

from __future__ import annotations

import os
import shutil
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
    import mol_study_concepts as msc
    from benchmark_corr_table import write_corr_tsv as write_benchmark_corr
    from benchmark_corr_table import write_facet_figure as write_benchmark_facet
    from concepts import build_table as build_concepts
    from concreteness import build_table as build_concreteness
    from concreteness import load_setup as load_concreteness_setup
    from energy_probe_mae_heatmap import plot_mae_heatmap as write_energy_probe
    from fair_pair_tables import main as write_fair_tables
    from legend_common import main as write_common_legend
    from plot_dataset_diversity_panels import write_figures as write_diversity
    from plot_decoy_vs_release import collect_and_write as write_decoy_vs_release
    from plot_eqhb_vs_benchmarks import main as write_eqhb_bench
    from plot_equal_hbond_vs_release import collect_and_write as write_equal_hbond
    from plot_favorable_vs_release import collect_and_write as write_favorable
    from plot_tau_vs_release_square import collect_and_write as write_tau_square
    from seed_spread_heatmap import collect_and_write as write_seed_heatmap
    from tau_floor2_table import collect_and_write as write_floor2_table
    from terms_old_neutral import main as write_terms_old_neutral

    from viz.three_mol_rows import main as write_mol_grid

    def _concreteness_table():
        models, mol_to_label, mol_labels = load_concreteness_setup()
        return build_concreteness(models, mol_to_label, mol_labels)

    def _mol_study_figures():
        d, order, _mt = ms.load_prepared(ms.DEFAULT_DATA)
        return [
            ms.plot_full(d, order, ms.DEFAULT_OUTDIR),
            ms.plot_square_model(d, ms.BEST_MODELS[0], ms.DEFAULT_OUTDIR,
                                 "ConcretenessCRA_GPT6_square.png",
                                 title="GPT-6 Astra", ylabel=False),
            ms.plot_square_model(d, ms.BEST_MODELS[1], ms.DEFAULT_OUTDIR,
                                 "ConcretenessCRA_GPT5_square.png", title="GPT-5"),
            ms.plot_all_models_horizontal(d, order, ms.DEFAULT_OUTDIR,
                                          "ConcretenessCRA_all_horizontal.png",
                                          lw=1.5, title="All 28 models"),
        ]

    def _paired_vs_uff():
        subprocess.run(
            [sys.executable, "metrics/run_paired_vs_uff.py"],
            check=True,
        )

    def _cohort_tables():
        out = []
        for name in ("models_table.tsv", "models_reasoning_params.tsv"):
            dst = PROJECT_ROOT / "figures" / name
            shutil.copyfile(PROJECT_ROOT / name, dst)
            out.append(dst)
        return out

    steps: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "all-pairs CRA vs release (supplementary AP panel)",
            lambda: write_tau_square(
                out=PROJECT_ROOT / "figures" / "cra_allpairs_vs_release.png",
                with_legend=False,
            ),
        ),
        ("equal-HB CRA vs benchmarks", write_eqhb_bench),
        ("equal-HB benchmark correlations (R^2 table) + facet", write_benchmark_corr),
        ("equal-HB CRA vs all benchmarks (facet)", write_benchmark_facet),
        ("fair CRA vs release", write_equal_hbond),
        ("unconventional CRA vs release", write_decoy_vs_release),
        ("conventional CRA vs release", write_favorable),
        ("common CRA legend", write_common_legend),
        ("seed spread heatmap (all models, min@10, equal-HB pairs)", write_seed_heatmap),
        ("CRA @ 2.0 table (mean / median / SD)", write_floor2_table),
        ("dataset diversity panels + ensemble_flexibility.tsv", write_diversity),
        ("energy probe MAE heatmap", write_energy_probe),
        ("concreteness table (explanation-level counts)", _concreteness_table),
        ("ConcretenessCRA + FullPlot", _mol_study_figures),
        ("concept counts table (explanation-level)", lambda: build_concepts()),
        ("FullPlotConcepts (concept facet)", lambda: msc.build_and_save()),
        ("27-molecule grid (PNG+SVG)", write_mol_grid),
        ("paired vs UFF TSV", _paired_vs_uff),
        ("prompt-vocabulary ablation TSVs (Muse Glimmer 30B)", write_terms_old_neutral),
        ("cohort tables (identities / reasoning+params)", _cohort_tables),
        ("fair-pair tables (*_fp.tsv)", write_fair_tables),
    )
    for label, run in steps:
        _run_step(label, run)
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
