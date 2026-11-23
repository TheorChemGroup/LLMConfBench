#!/usr/bin/env python3

"""Write the ``*_fp.tsv`` fair-pair versions of the analysis tables.

``first_plots.py`` runs this after the all-pairs and explanation-table steps.
Fair pairs are the 2,997 reliable pairs whose two conformers have the same
number of intramolecular H-bonds.
"""

from __future__ import annotations

import csv
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
    load_predictions,
    load_r2scan,
    override_r2scan3c,
)
from terms_old_neutral import aggregate_by_molecule
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
from metrics.grouped_metrics import fmt_p, summary_stats
from metrics.option2_metrics import load_valid_answers
from metrics.paired_vs_baseline import summarize_llm_vs_baseline

FIG = _REPO / "figures"
SEED_DIRS = [_REPO / "geom-qm9" / b for b in BUNDLES]
PROMPTS = {
    "old": SEED_DIRS,
    "neutral": [_REPO / "geom-qm9-neutral" / b for b in BUNDLES],
}
SEED_BY_BUNDLE = {
    "prompt_30x30_NEW": "42",
    "prompt_30x30_NEW_43": "43",
    "prompt_30x30_NEW_44": "44",
}
TAG = "muse-glimmer30b"


class _Context:
    def __init__(self) -> None:
        self.r2 = load_r2scan()
        self.molecules = sorted(bench_molecules())
        self.eq = equal_hbond_pair_tables(self.r2, hb_count_map(self.r2, self.molecules))
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
    rows.sort(key=lambda row: -row[1])

    path = FIG / "cra_floor2_table_fp.tsv"
    lines = ["model\tmean_tau_f2.0\tmedian_tau_f2.0\tsd_tau_f2.0\tn_molecules"]
    lines += [f"{name}\t{mean:.3f}\t{median:.3f}\t{sd:.3f}\t{n}"
              for name, mean, median, sd, n in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")
    return path


def write_paired_table(ctx: _Context) -> Path:
    uff = ctx.ff_taus(get_uff_ranking)
    rows = []
    for display, filename in model_files_by_display("geom_benchmark").items():
        taus = ctx.model_taus(filename)
        common = sorted(set(taus) & set(uff))
        if not common:
            continue
        summary = summarize_llm_vs_baseline(
            [taus[m] for m in common],
            [uff[m] for m in common],
            n_perm=10_000,
            seed=0,
        )
        rows.append((short_label_for_display(display), summary))
    rows.sort(key=lambda row: -row[1].mean)

    path = FIG / "paired_llm_minus_uff_floor2_fp.tsv"
    lines = ["model\tmean_d\tmedian_d\tci_lo\tci_hi\tperm_pvalue\twilcoxon_pvalue"]
    for label, summary in rows:
        lines.append(
            "\t".join([
                label,
                f"{summary.mean:.3f}",
                f"{summary.median:.3f}",
                f"{summary.ci_lo:.3f}",
                f"{summary.ci_hi:.3f}",
                fmt_p(summary.perm_pvalue, 3),
                fmt_p(summary.wilcoxon_pvalue, 3),
            ])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")
    return path


def write_terms_tables(ctx: _Context) -> list[Path]:
    source = FIG / "terms_old_neutral_museglimmer30b.tsv"
    with source.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    fieldnames = list(rows[0])

    predictions = {}
    metas = {}
    for prompt, dirs in PROMPTS.items():
        for bundle in dirs:
            seed = SEED_BY_BUNDLE[bundle.name]
            metas[(prompt, seed)] = load_meta(bundle, ctx.override)
            predictions[(prompt, seed)] = load_predictions(
                bundle, f"predictions_{TAG}.jsonl"
            )

    fair_rows = []
    for row in rows:
        prompt = row["prompt"]
        seed = str(row["seed"])
        mol = row["molecule"]
        tau = equal_hb_cra(
            ctx.eq.get(mol, []),
            metas[(prompt, seed)][mol]["public_label_to_conf_index"],
            predictions[(prompt, seed)].get(mol, []),
        )
        fair = dict(row)
        fair["tau_f2.0"] = "" if tau is None else f"{tau:.3f}"
        fair_rows.append(fair)

    per_row = FIG / "terms_old_neutral_museglimmer30b_fp.tsv"
    with per_row.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(fair_rows)

    count_cols, agg_rows = aggregate_by_molecule(fair_rows)

    per_molecule = FIG / "terms_old_neutral_museglimmer30b_agg_fp.tsv"
    agg_fields = ["prompt", "label", "molecule", "n_seeds", "tau_f2.0", "words"]
    agg_fields += count_cols
    with per_molecule.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Saved {per_row}")
    print(f"Saved {per_molecule}")
    return [per_row, per_molecule]


def main() -> int:
    ctx = _Context()
    write_cra_table(ctx)
    write_paired_table(ctx)
    write_terms_tables(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
