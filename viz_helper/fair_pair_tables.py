#!/usr/bin/env python3

"""Write the ``*_fp.tsv`` fair-pair versions of the analysis tables.

``first_plots.py`` runs this after the all-pairs and explanation-table steps.
Fair pairs are the 2,997 reliable pairs whose two conformers have the same
number of intramolecular H-bonds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_REPO), str(_REPO / "viz_helper")]

from decoy_common import (
    equal_hb_cra,
    BUNDLES,
    bench_molecules,
    equal_hbond_pair_tables,
    hb_count_map,
    load_meta,
    load_r2scan,
    override_r2scan3c,
)
from plot_metrics_vs_release import (
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
)

from config.models_config import (
    geom_conformers_dir,
    model_files_by_display,
    short_label_for_display,
)
from metrics.grouped_metrics import summary_stats
from metrics.option2_metrics import load_valid_answers
from viz_helper.hb_rule_baseline import per_molecule

FIG = _REPO / "figures"
SEED_DIRS = [_REPO / "geom-qm9" / b for b in BUNDLES]


class _Context:
    def __init__(self) -> None:
        self.r2 = load_r2scan()
        self.molecules = sorted(bench_molecules())
        self.counts = hb_count_map(self.r2, self.molecules)
        self.eq = equal_hbond_pair_tables(self.r2, self.counts)
        self.override = override_r2scan3c()
        self.metas = {d: load_meta(d, self.override) for d in SEED_DIRS}

    def model_taus(self, filename: str) -> dict[str, float]:
        answers = {d: load_valid_answers(d / filename, self.metas[d]) for d in SEED_DIRS}
        out = {}
        for mol in self.molecules:
            vals = []
            for d in SEED_DIRS:
                if mol not in answers[d]:
                    continue
                tau = equal_hb_cra(
                    self.eq.get(mol, []),
                    self.metas[d][mol]["public_label_to_conf_index"],
                    answers[d][mol],
                )
                if tau is not None:
                    vals.append(tau)
            if len(vals) == len(SEED_DIRS):
                out[mol] = float(np.mean(vals))
        return out

    def ff_taus(self, getter) -> dict[str, float]:
        meta = self.metas[SEED_DIRS[0]]
        out = {}
        for mol in self.molecules:
            public_map = meta[mol]["public_label_to_conf_index"]
            conf_to_public = {int(i): str(label) for label, i in public_map.items()}
            ranking = getter(geom_conformers_dir() / mol)
            if not ranking:
                continue
            prediction = [conf_to_public[i] for i in ranking if i in conf_to_public]
            tau = equal_hb_cra(self.eq.get(mol, []), public_map, prediction)
            if tau is not None:
                out[mol] = tau
        return out


def write_cra_table(ctx: _Context) -> Path:
    rows = []
    for display, filename in model_files_by_display("geom_benchmark").items():
        taus = ctx.model_taus(filename)
        if taus:
            mean, median, sd, n = summary_stats(list(taus.values()))
            rows.append((short_label_for_display(display), mean, median, sd, n))
    for name, getter in (
        ("UFF", get_uff_ranking),
        ("GFN-FF", get_gfnff_ranking),
        ("MMFF94", get_mmff94_ranking),
    ):
        taus = ctx.ff_taus(getter)
        if taus:
            mean, median, sd, n = summary_stats(list(taus.values()))
            rows.append((name, mean, median, sd, n))
    hb = per_molecule(ctx.eq, ctx.counts)
    if hb:
        rows.append(("HBC", *summary_stats(hb)))
    rows.sort(key=lambda row: -row[1])

    path = FIG / "cra_floor2_table_fp.tsv"
    lines = ["model\tmean_tau_f2.0\tmedian_tau_f2.0\tsd_tau_f2.0\tn_molecules"]
    lines += [f"{name}\t{mean:.3f}\t{median:.3f}\t{sd:.3f}\t{n}"
              for name, mean, median, sd, n in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")
    return path


def main() -> int:
    ctx = _Context()
    write_cra_table(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
