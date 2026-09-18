# Molecular Geometry Understanding Has Unintendedly Emerged in Frontier Large Language Models

_Gregorii A. Semakin\*, Timofey V. Losev, Ilya V. Prolomov, Stepan N. Ostarkov, Igor V. Alabugin, Michael G. Medvedev\*_

\* Corresponding authors.

Code and data for **LLMConfBench**, the benchmark behind the paper above. A model receives the XYZ coordinates of 30 conformers of a single molecule and must return their stability ranking; scores are reliable-pair Kendall's tau computed against **r2SCAN-3c** reference energies. UFF, MMFF94 and GFN-FF are the force-field baselines, and DLPNO-CCSD(T) provides an independent check of the reference.

---

## Installation

```bash
git clone https://github.com/TheorChemGroup/LLMConfBench
cd LLMConfBench

conda env create -f environment.yml
conda activate llm_benchmark
```

Copy `.env.template` to `.env` and set `OPENROUTER_API_KEY` if you will re-run LLM inference. Models served locally through Ollama (exact identifiers such as `gemma3:27b`, listed in `models_table.tsv`) additionally require a running Ollama server (`OLLAMA_HOST`, default `127.0.0.1:11434`).

---

## Reproduce paper plots

`viz/first_plots.py` is the **single entry point** and regenerates every manuscript figure/table:

```bash
python viz_helper/build_texts.py
python viz/first_plots.py
```

It writes:

- `figures/release_scatter/r2scan3c/` — `tau_vs_release_square_w95_2.png`, `tau_vs_params_square_w95_2.png`, `tau_vs_cost_square_w95_2.png`, `tau_vs_gpqa_square_w95_2.png`, `tau_vs_scicode_square_w95_2.png`, `tau_vs_arcagi1_square_w95_2.png`, `tau_vs_arcagi2_square_w95_2.png`, `tau_vs_benchmarks_facet_2.png`, plus `tau_benchmark_correlations_2.tsv` (all benchmark R²/ρ predictors)
- `figures/r2scan3c/` — `molecules_27_3row.{png,svg}`, `seed_spread_heatmap.png`, `ConcretenessTau.png`, `FullPlot.png`, `energy_probe_mae_heatmap.png`, `tanimoto_distance_distr.png`, `ensemble_rmsd_tfd_vs_nrot.png`, `deltae_energy_profiles.png`, `deltae_pair_diff_profiles.png`, `tau_floor2_table.{png,tsv}`, `ensemble_flexibility.tsv`, and `paired_llm_minus_uff_floor2.tsv`

Two intermediate tables are written alongside them: `seed_spread_heatmap_molecules.tsv` (molecule IDs, SMILES and τ in heatmap order) and `concreteness_vs_tau.tsv` (explanation-level H-bond/word counts).

### Inputs required

The pipeline reads a fixed set of inputs from the working tree:

| Input | Used for |
|-------|----------|
| `geom-qm9/output_30_confs/<mol>/*.xyz` | geometries of molecules |
| `geom-qm9/prompt_30x30_NEW{,_43,_44}/` | prompts, answers and predictions (3 seeds) |
| `geom-qm9/energy_probe_*.jsonl` | energy-recall test |
| `texts/seed{42,43,44}/<model>.jsonl` | per-run explanations (H-bond/concreteness analysis; rebuild with `python viz_helper/build_texts.py`) |
| `orca_r2scan-3c/deltaE_r2scan3c_kcal_mol.json` | r2SCAN-3c reference energies |
| `models.yaml`, `models_table.tsv` | cohort, tags, release dates |
| `cohort_benchmark_scores.json`, `arcagi.json` | GPQA / SciCode / ARC-AGI scores |
| `MDs/MOLS.md` | SMILES topology for UFF/MMFF94 |

---

## Reproduce article results

### A. Geometries (from GEOM-QM9)

Run `llm_rerank.ipynb` only once. It downloads GEOM, samples 30 molecules with ≥30 conformers from the QM9 subset as we did, and writes:

`geom-qm9/output_30_confs/<mol>/<mol>_conf_0..29.xyz`

The repo already contains the frozen XYZ set used in the paper; re-running the notebook with a new shuffle yields a **different** molecule list.

### B. Energy baselines

```bash
# r2SCAN-3c reference energies
python orca_r2scan-3c/make_orca_r2scan3c_inputs.py
# run ORCA on orca_r2scan-3c/orca_r2scan3c/**/*.inp
python orca_r2scan-3c/parse_orca_energies.py

# 10-molecule DLPNO-CCSD(T) subset — keep subset_molecules.txt as committed
python orca_ccsdt/make_orca_dlpno_inputs.py
# run ORCA on orca_ccsdt/orca_dlpno_ccsdt/**/*.inp
python orca_ccsdt/parse_orca_dlpno.py
```

Do **not** re-run `orca_ccsdt/select_subset.py` if you want the published subset.

### Regenerate DFT / CCSD(T) comparison reports

Committed energy tables:

| Method | File |
|--------|------|
| r2SCAN-3c (32 molecules: 30 GEOM-QM9 + 2 de-novo) | `orca_r2scan-3c/deltaE_r2scan3c_kcal_mol.json` |
| DLPNO-CCSD(T) (10-molecule subset) | `orca_ccsdt/deltaE_dlpno_ccsdt_kcal_mol.json` |

Then make text comparisons:

```bash
# xTB vs r2SCAN-3c vs CCSD(T) agreement -> orca_ccsdt/xtb_vs_r2scan_vs_ccsdt.txt
python orca_ccsdt/compare_xtb_ccsdt.py

# r2SCAN vs CCSD(T) pair agreement + model-ranking Spearman/Kendall -> orca_ccsdt/compare_baselines_subset.txt
python orca_ccsdt/compare_baselines_subset.py
```

If you re-ran ORCA, refresh the JSONs first:

```bash
python orca_r2scan-3c/parse_orca_energies.py
python orca_ccsdt/parse_orca_dlpno.py
```

Raw ORCA `.out` files are only present after ORCA runs, e.g. `orca_r2scan-3c/orca_r2scan3c/<mol>/<mol>_conf_0.out`.

### C. Prompts (3 seeds)

Each bundle `geom-qm9/prompt_30x30_NEW{,_43,_44}` contains the exact text shown to the models: 30 conformer blocks per molecule, with the label -> conformer mapping and the block order shuffled per seed. The committed bundles are the ones the recorded `answers_*` / `predictions_*` were collected with, so **do not regenerate them unless you intend to re-run every model**.

`build_prompts.py` consumes a single `random.Random(seed)` for the whole invocation. Each molecule's label permutation and block order therefore depend on how many molecules were processed before it, i.e. on the exact set **and order** of the input folders.

To keep the committed bundles reproducible, they were built from **three separate invocations per seed**:

1. one run over the **original 30 GEOM-QM9 molecules** (sorted, *excluding* the two de-novo pilots) — this run yields the 25 scored GEOM-QM9 molecules (and the 5 dropped ones), and
2. **one run per pilot**, i.e. each pilot processed on its own.

Consequently, a single run over today's `output_30_confs` (32 folders) or over just the 27-molecule cohort produces different shuffles.

`prompts/rebuild_prompt_bundles.py` performs exactly this recipe and merges the per-run metadata into the committed 27-row `prompt_metadata.jsonl`:

```bash
python prompts/rebuild_prompt_bundles.py --out-dir geom-qm9   # rewrites the three bundles
```

This reproduces the committed prompts byte-for-byte: 32 `.txt` per bundle (30 GEOM-QM9 + 2 pilots) and 27 `prompt_metadata.jsonl` rows (25 scored GEOM-QM9 molecules + 2 pilots). The 5 GEOM-QM9 molecules dropped from the scored cohort keep their `.txt` in the bundle but are absent from `prompt_metadata.jsonl`, which is the file the runners iterate.

### D. LLM rankings

```bash
python runners/run_prompt_bundle_openrouter.py \
  --bundle-dir geom-qm9/prompt_30x30_NEW \
  --model <openrouter-id> \
  --output-tag <tag>
```

Repeat for seeds 43 and 44. Local models: `runners/run_ollama.py` with the same flags. Tags must match `answers:` in `models.yaml`.

Then regenerate figures with `python viz/first_plots.py`.

### E. Adding a new model or molecule

**Model.** To add a model to the cohort:

1. Collect all three prompt presentations (one response per molecule × seed) and store them as `answers_<tag>.jsonl` / `predictions_<tag>.jsonl` in `geom-qm9/prompt_30x30_NEW{,_43,_44}/`:

   ```bash
   python runners/run_prompt_bundle_openrouter.py \
     --bundle-dir geom-qm9/prompt_30x30_NEW --model <openrouter-id> --output-tag <tag>
   # repeat for geom-qm9/prompt_30x30_NEW_43 and _44
   # local models: runners/run_ollama.py with the same flags
   ```

2. Register it in `models.yaml` (keys must be unique; `answers` must match the tag):

   ```yaml
   mymodel:
     display: My Model            # shown in tables and figures
     short: my-model              # slug used for external-benchmark matching
     answers: answers_mymodel.jsonl
     color: "#6b7280"
     class: closed                # open-weight models use `os`
     reasoning: true
     release_date: "2026-01-01"
     params_total_b: 70
     params_active_b: 70          # equal to total for dense models
     arch: dense                  # dense | moe | distill
   ```

3. Add its id to `groups.geom_benchmark` in `models.yaml` so it enters the cohort.
4. Optional — external benchmark scores: add the model to `config/cohort_model_slugs.py` and, for OpenRouter matching, to `OPENROUTER_SLUG_CANDIDATES` in `viz_helper/refresh_cohort_benchmark_scores.py`; provider presets for new OpenRouter models go in `runners/api_client.py`.
5. A model is only included when **all three seeds are complete** (`viz/cohort.py` drops incomplete models with a warning).
6. Regenerate everything: `python viz/first_plots.py`.

**Molecule.** To add a molecule to the benchmark:

1. Add 30 conformers as `geom-qm9/output_30_confs/<mol>/<mol>_conf_0..29.xyz` (atom order identical across the 30 files; the `Energy:` comment line is optional).
2. Add its SMILES to `MDs/MOLS.md` (the folder name is the sanitized SMILES); this provides the UFF/MMFF94 bonded topology.
3. Add reference energies to `orca_r2scan-3c/deltaE_r2scan3c_kcal_mol.json` as `"<mol>": {"0": 0.0, ...}` with 30 conformer-index keys.
4. Rebuild the prompt bundles — note `build_prompts.py` draws from a single `random.Random(seed)` per run, so to leave existing prompts untouched, follow the exact recipe in section C or add the new molecule in its own run.
5. The molecule enters the scored cohort only if it has **≥15 conformer pairs with |ΔE| ≥ 2.0 kcal/mol**; otherwise it is dropped.

---

## Repository structure

```
llm/
├── environment.yml         # env
├── models.yaml             # models, paths, baselines, cohort
├── models_table.tsv        # curated release/access dates
├── cohort_benchmark_scores.json  # GPQA / SciCode / ARC-AGI scores
├── arcagi.json             # ARC-AGI-1/2 raw scores
├── llm_rerank.ipynb        # GEOM -> output_30_confs
├── config/                 # load models.yaml helpers
├── metrics/                # Kendall τ, reliable-pair floors, min@k, paired vs UFF, date trend
├── prompts/                # build shuffled conformer prompts (build_prompts.py, rebuild_prompt_bundles.py)
├── runners/                # OpenRouter / Ollama inference + energy-recall test
├── test/                   # unit tests (`python -m unittest discover -s test -v`)
├── viz/                    # first_plots.py (entry point), three_mol_rows.py (27-grid),
│                           # plot_style.py, cohort.py
├── viz_helper/             # τ plots, seed-spread heatmap, diversity + energy-recall test panels,
│                           # concreteness analysis, shared scoring/force-field helpers
├── geom-qm9/
│   ├── output_30_confs/    # XYZ geometries (30 GEOM-QM9 molecules + 2 de-novo, 30 confs each)
│   ├── prompt_30x30_NEW{,_43,_44}/   # prompts + answers + predictions (paper: 27 mols)
│   └── energy_probe_*.jsonl          # energy-recall test responses
├── texts/                  # per-run explanations used for the H-bond/concreteness analysis
├── orca_r2scan-3c/         # r2SCAN-3c inputs/parse + deltaE JSON
├── orca_ccsdt/             # DLPNO-CCSD(T) subset + deltaE JSON
├── figures/                # manuscript figures (regenerated by first_plots)
└── MDs/MOLS.md             # SMILES list (molecule grid + UFF/MMFF topology)
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you use this code or dataset, please cite the accompanying paper:

```bibtex
@misc{semakin2026moleculargeometryunderstandingunintendedly,
      title={Molecular Geometry Understanding Has Unintendedly Emerged in Frontier Large Language Models},
      author={Gregorii A. Semakin and Timofey V. Losev and Ilya V. Prolomov and Stepan N. Ostarkov and Igor V. Alabugin and Michael G. Medvedev},
      year={2026},
      eprint={2609.20666},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2609.20666},
}
```

---

## Contact

- Gregorii A. Semakin: gregoriisemakin@gmail.com
- Michael G. Medvedev: medvedev.m.g@gmail.com
