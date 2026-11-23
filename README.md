## Installation

```bash
conda env create -f environment.yml
conda activate llmconfbench
```

## Figures

```bash
python viz_helper/build_texts.py   # explanation texts (texts/, not tracked)
python viz/first_plots.py          # rebuilds pair_runs_reliable.csv, then figures/ and tables
```

`pair_runs_reliable.csv` is derived (model answers + r2SCAN-3c energies + MMFF94 terms,
H-bonds at H···A <= 2.5 Å and D–H···A >= 120°), so it is rebuilt by the first step and is
not tracked. `viz_helper/build_pair_table.py` can still be run on its own.

Main outputs in `figures/`: `cra_allpairs_vs_release.png`, `cra_no_hbond_vs_release.png`,
`cra_conventional_vs_release.png`, `cra_unconventional_vs_release.png`,
`concreteness_{gpt5,gpt6,opus5}.png`, `fig1c_amplified_term_contributions.png`,
`fig3a_decoy_gallery.png`, `fig3b_quotes.png`.

## Citation

```bibtex
@misc{semakin2026humanlanguageshapesemerging,
  title={Human Language Shapes the Emerging Physical Reasoning in Large Language Models},
  author={Gregorii A. Semakin and Timofey V. Losev and Ilya V. Prolomov and Stepan N. Ostarkov and Igor V. Alabugin and Michael G. Medvedev},
  year={2026},
  eprint={2609.20666},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2609.20666}
}
```

## License

MIT — see `LICENSE`.
