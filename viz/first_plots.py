#!/usr/bin/env python3

"""Generate the manuscript figures and tables (one command).

    python viz/first_plots.py

Produces:
  pair_runs_reliable.csv                (rebuilt from the model answers)
  figures/molecules_27_3row.png
  figures/cra_allpairs_vs_release.png   (all-pairs metric)
  figures/cra_vs_arcagi1.png            (equal-HB metric)
  figures/cra_vs_cost.png               (equal-HB metric)
  figures/cra_vs_gpqa.png               (equal-HB metric)
  figures/cra_vs_scicode.png            (equal-HB metric)
  figures/cra_vs_benchmarks_facet.png   (equal-HB metric)
  figures/cra_unconventional_vs_release.png
  figures/cra_conventional_vs_release.png
  figures/seed_spread_heatmap.png
  figures/tanimoto_distance_distr.png
  figures/ensemble_rmsd_tfd_vs_nrot.png
  figures/deltae_energy_profiles.png
  figures/deltae_pair_diff_profiles.png
  figures/energy_probe_mae_heatmap.png
  figures/recovery_pooled_fit8_normalised_above_uff.png
  mmff94_terms_by_pair.csv
  figures/cra_no_hbond_vs_release.png
  figures/fig3a_decoy_gallery.png
  figures/fig3b_quotes.png (+ .json)
  figures/ed_mmff94_alignment_scatter.png
  figures/concreteness_{gpt5,gpt6,opus5}.png  (Fig. 2d-f)
  figures/concreteness_full.png
  figures/FullPlotConcepts.png
  figures/ensemble_flexibility.tsv
  figures/cra_floor2_table.tsv
  figures/paired_llm_minus_uff_floor2.tsv
  figures/mmff94_alignment_per_molecule.tsv
  figures/terms_old_neutral_museglimmer30b.tsv (+ _agg.tsv)
  figures/models_table.tsv
  figures/models_reasoning_params.tsv
  figures/*_fp.tsv                        (fair-pair variants)

Intermediate tables written by the same steps: concreteness_vs_cra_fp.tsv,
seed_spread_heatmap_molecules_fp.tsv, concept_counts_vs_cra_fp.tsv.

Fig. 3a assets are rendered with xyzrender by render_decoy_gallery.py within
the decoy-gallery step.
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
    import mol_study_concepts as msc
    from benchmark_corr_table import write_facet_figure as write_benchmark_facet
    from cohort_tables import main as write_cohort_tables
    from concepts import build_table as build_concepts
    from concreteness import build_table as build_concreteness
    from concreteness import load_setup as load_concreteness_setup
    from concreteness_panel import main as write_concreteness_panels
    from decoy_gallery import main as write_decoy_gallery
    from energy_probe_mae_heatmap import plot_mae_heatmap as write_energy_probe
    from fair_pair_tables import main as write_fair_tables
    from mmff_alignment import main as write_mmff_alignment
    from plot_dataset_diversity_panels import write_figures as write_diversity
    from plot_decoy_vs_release import collect_and_write as write_decoy_vs_release
    from plot_eqhb_vs_benchmarks import main as write_eqhb_bench
    from plot_favorable_vs_release import collect_and_write as write_favorable
    from plot_tau_vs_release_square import collect_and_write as write_tau_square
    from quotes_panel import main as write_quotes
    from render_decoy_gallery import main as render_decoy_gallery
    from seed_spread_heatmap import collect_and_write as write_seed_heatmap
    from tau_floor2_table import collect_and_write as write_floor2_table
    from plot_no_hbond_vs_release import collect_and_write as write_no_hbond
    from recovery_above_uff import main as write_recovery_uff
    from terms_old_neutral import main as write_terms_old_neutral

    from viz.three_mol_rows import main as write_mol_grid

    def _concreteness_table():
        models, mol_to_label, mol_labels = load_concreteness_setup()
        return build_concreteness(models, mol_to_label, mol_labels)

    def _mol_study_figures():
        d, order, _mt = ms.load_prepared(ms.DEFAULT_DATA)
        return [ms.plot_full(d, order, ms.DEFAULT_OUTDIR)]

    def _mmff94_terms():
        subprocess.run(
            [sys.executable, "viz_helper/mmff94_terms.py"],
            check=True,
        )
        return PROJECT_ROOT / "mmff94_terms_by_pair.csv"

    def _pair_table():
        subprocess.run(
            [sys.executable, "viz_helper/build_pair_table.py"],
            check=True,
        )
        return PROJECT_ROOT / "pair_runs_reliable.csv"

    def _paired_vs_uff():
        subprocess.run(
            [sys.executable, "metrics/run_paired_vs_uff.py"],
            check=True,
        )

    def _decoy_gallery():
        render_decoy_gallery()
        return write_decoy_gallery()

    steps: tuple[tuple[str, Callable[[], Any]], ...] = (
        # Writes figures/seed_spread_heatmap_molecules_fp.tsv (the molecule
        # numbering used by the term table and the pair table), so it must run
        # before both.
        ("seed spread heatmap (all models, min@10, equal-HB pairs)", write_seed_heatmap),
        ("MMFF94 term decomposition (mmff94_terms_by_pair.csv)", _mmff94_terms),
        ("pair table rebuild (pair_runs_reliable.csv)", _pair_table),
        (
            "all-pairs CRA vs release (supplementary AP panel)",
            lambda: write_tau_square(
                out=PROJECT_ROOT / "figures" / "cra_allpairs_vs_release.png",
                with_legend=False,
            ),
        ),
        ("no-H-bond CRA vs release (Fig. 2)", write_no_hbond),
        ("equal-HB CRA vs benchmarks", write_eqhb_bench),
        ("equal-HB CRA vs all benchmarks (facet)", write_benchmark_facet),
        ("unconventional CRA vs release", write_decoy_vs_release),
        ("conventional CRA vs release", write_favorable),
        ("CRA @ 2.0 table (mean / median / SD)", write_floor2_table),
        ("dataset diversity panels + ensemble_flexibility.tsv", write_diversity),
        ("energy probe MAE heatmap", write_energy_probe),
        ("concreteness table (explanation-level counts)", _concreteness_table),
        ("concreteness_full (28-panel facet)", _mol_study_figures),
        ("concreteness panels (Fig. 2d-f)", write_concreteness_panels),
        ("concept counts table (explanation-level)", lambda: build_concepts()),
        ("FullPlotConcepts (concept facet)", lambda: msc.build_and_save()),
        ("27-molecule grid (PNG+SVG)", write_mol_grid),
        ("paired vs UFF TSV", _paired_vs_uff),
        ("prompt-vocabulary ablation TSVs (Muse Glimmer 30B)", write_terms_old_neutral),
        ("cohort tables from models.yaml (identities / reasoning+params)", write_cohort_tables),
        ("fair-pair tables (*_fp.tsv)", write_fair_tables),
        ("MMFF94-r2SCAN alignment (ED S9)", write_mmff_alignment),
        ("term recovery above UFF (Fig. 1c, normalised)", write_recovery_uff),
        ("decoy gallery render + assembly (Fig. 3a)", _decoy_gallery),
        ("quotes panel (Fig. 3b)", write_quotes),
    )
    for label, run in steps:
        _run_step(label, run)
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
