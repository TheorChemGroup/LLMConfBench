"""CRA on equal-H-bond pairs vs external benchmarks.

y = mean per-molecule CRA on the 2,997 pairs with equal H-bond counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
REPO = _REPO
sys.path[:0] = [str(REPO), str(REPO / "viz_helper")]

from benchmark_scores import load_cohort_benchmark_scores
from decoy_common import (
    BUNDLES,
    bench_molecules,
    equal_hbond_pair_tables,
    hb_count_map,
    score_pairs_ff,
    score_pairs_letters,
)
from plot_equal_hbond_vs_release import _macro
from plot_metrics_vs_release import (
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
    total_api_cost_by_display,
)
from plot_tau_vs_arcagi import DEFAULT_ARCAGI, load_arc_by_display
from plot_tau_vs_benchmark import GPQA_TREND_EXCLUDE, write_tau_vs_aa
from plot_tau_vs_release_square import write_square

from config.models_config import (
    geom_seed_dirs,
    model_files_by_display,
    resolve_deltae_json,
)
from metrics.kendall_tau_ranking import load_deltae_override, load_metadata_index

GEOM = REPO / "geom-qm9"
OUT = REPO / "figures"
_EQUAL_HB_CACHE: dict[str, tuple] = {}


def baseline_energies(baseline: str):
    """Per-conformer energies for ``baseline`` and the matching override dict."""
    override = load_deltae_override(resolve_deltae_json(baseline))
    if override is not None:
        energies = {
            mol: {str(ci): float(value) for ci, value in values.items()}
            for mol, values in override.items()
        }
        return energies, override

    # GFN2-xTB has no override file; its energies live in the prompt metadata.
    meta = load_metadata_index(
        geom_seed_dirs()[0][1] / "prompt_metadata.jsonl"
    )
    energies = {}
    for mol, row in meta.items():
        public = row.get("public_label_to_conf_index", {})
        deltae = row.get("letter_to_deltaE_kcal_mol", {})
        energies[mol] = {
            str(int(conf)): float(deltae[label])
            for label, conf in public.items()
            if label in deltae
        }
    return energies, None


def equal_hb_rows(baseline: str = "r2scan3c"):
    if baseline in _EQUAL_HB_CACHE:
        return _EQUAL_HB_CACHE[baseline]

    energies, override = baseline_energies(baseline)
    counts = hb_count_map(energies, bench_molecules())
    table = equal_hbond_pair_tables(energies, counts)
    n_pairs = sum(len(v) for v in table.values())
    bdirs = [GEOM / b for b in BUNDLES]

    rows = []
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            continue
        st = score_pairs_letters(bdirs, pred, {"eq": table}, override)["eq"]
        mean, half, n = _macro(st)
        rows.append({
            "name": display,
            "eqhb": mean,
            "eqhb_ci": half,
            "n": n,
            "pilot_partial": n < 27,
        })

    ff = {}
    for name, getter in (("UFF", get_uff_ranking), ("GFN-FF", get_gfnff_ranking),
                         ("MMFF94", get_mmff94_ranking)):
        st = score_pairs_ff(getter, {"eq": table}, energies)["eq"]
        mean, half, n = _macro(st)
        ff[name] = {"eqhb": mean, "eqhb_ci": half, "n": n}

    ff_bits = [(n, f"{v['eqhb']:+.3f}") for n, v in ff.items()]
    print(
        f"equal-H-bond pairs ({baseline}): {n_pairs}, "
        f"models: {len(rows)}, FF: {ff_bits}"
    )
    _EQUAL_HB_CACHE[baseline] = (rows, ff, n_pairs, override)
    return _EQUAL_HB_CACHE[baseline]


def cost_scores(override) -> dict[str, float]:
    costs, _skipped, _mols = total_api_cost_by_display(deltae_override=override)
    return costs


def bench_scores(bench: str) -> dict[str, float]:
    scores = load_cohort_benchmark_scores()
    out = {}
    for name, vals in scores.items():
        val = vals.get(bench)
        if val is not None:
            out[name] = float(val)
    return out


def arc_scores(key: str) -> dict[str, float]:
    arc = load_arc_by_display(DEFAULT_ARCAGI)
    return {name: vals[key] for name, vals in arc.items() if key in vals}


def main() -> None:
    rows, ff, _n_pairs, override = equal_hb_rows("r2scan3c")
    ylabel = "CRA across fair pairs"
    common = {
        "models": rows,
        "rand": {},
        "uff": ff["UFF"],
        "gfnff": ff["GFN-FF"],
        "mmff": ff["MMFF94"],
        "y_key": "eqhb",
        "ylabel": ylabel,
    }

    panels = [
        ("cost", "Total API cost ($)", cost_scores(override), None),
        ("gpqa", "GPQA Diamond", bench_scores("gpqa"), GPQA_TREND_EXCLUDE),
        ("scicode", "SciCode", bench_scores("scicode"), None),
        ("arcagi1", "ARC-AGI-1", arc_scores("arc_agi_1"), None),
        ("arcagi2", "ARC-AGI-2", arc_scores("arc_agi_2"), None),
    ]
    for stem, xlabel, scores, exclude in panels:
        dest = OUT / f"cra_vs_{stem}.png"
        write_tau_vs_aa(dest, scores_by_name=scores, xlabel=xlabel,
                        second_trend_exclude=exclude, **common)
        print(f"Saved {dest}")

    params_dest = OUT / "cra_vs_params.png"
    write_square(
        params_dest,
        rows,
        {},
        ff["UFF"],
        ff["GFN-FF"],
        ff["MMFF94"],
        {},
        y_key="eqhb",
        ylabel=ylabel,
        show_ci=True,
        x_mode="params",
        marker_by="reasoning",
        size_by_params=False,
        with_legend=True,
    )
    print(f"Saved {params_dest}")


if __name__ == "__main__":
    main()
