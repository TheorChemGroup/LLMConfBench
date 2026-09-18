#!/usr/bin/env python3

"""CLI: paired reliable-τ differences (LLM − UFF) on the geom cohort.

Writes a TSV by default and prints a summary table.

Example:
  python metrics/run_paired_vs_uff.py
  python metrics/run_paired_vs_uff.py --model "Claude Opus 5" --floor 2.0
  python metrics/run_paired_vs_uff.py --out figures/r2scan3c/paired_llm_minus_uff.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import (
    geom_conformers_dir,
    geom_seed_paths,
    model_files_by_display,
    resolve_deltae_json,
    short_label_for_display,
)
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import load_valid_answers, score_molecule
from metrics.paired_vs_baseline import PairedDiffSummary, summarize_llm_vs_baseline
from viz.cohort import complete_geom_models
from viz_helper.plot_metrics_vs_release import get_uff_ranking

DEFAULT_OUT = Path("figures/r2scan3c/paired_llm_minus_uff_floor2.tsv")
TSV_HEADER = (
    "model\tmean_d\tmedian_d\tci_lo\tci_hi\tperm_pvalue\twilcoxon_pvalue"
)


def _tau_for_ranking(letter_de: dict, ranking: list[str], floor: float) -> float | None:
    scored = score_molecule(ranking, letter_de, [floor])
    if not scored.get("valid"):
        return None
    value = scored.get(f"tau_f{floor}")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _uff_per_molecule(meta: dict, conformers_dir: Path, floor: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for mol, row in meta.items():
        folder = conformers_dir / mol
        pub = row.get("public_label_to_conf_index", {})
        letter_de = row.get("letter_to_deltaE_kcal_mol", {})
        if not folder.is_dir() or not pub or not letter_de:
            continue
        conf_to_pub = {int(v): str(k) for k, v in pub.items()}
        rank_idx = get_uff_ranking(folder)
        if not rank_idx:
            continue
        pred = [conf_to_pub[i] for i in rank_idx if i in conf_to_pub]
        tau = _tau_for_ranking(letter_de, pred, floor)
        if tau is not None:
            out[mol] = tau
    return out


def _llm_mean_per_molecule(
    filename: str,
    seed_metas: dict[Path, dict],
    molecules: list[str],
    floor: float,
) -> dict[str, float]:
    seed_dirs = list(seed_metas)
    out: dict[str, float] = {}
    for mol in molecules:
        vals: list[float] = []
        for seed_dir in seed_dirs:
            meta = seed_metas[seed_dir]
            answers = load_valid_answers(seed_dir / filename, meta)
            if mol not in answers:
                continue
            tau = _tau_for_ranking(meta[mol]["letter_to_deltaE_kcal_mol"], answers[mol], floor)
            if tau is not None:
                vals.append(tau)
        if len(vals) == len(seed_dirs):
            out[mol] = float(np.mean(vals))
    return out


def _fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.3f}"


def _fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.3f}"


def _tsv_row(label: str, summary: PairedDiffSummary) -> str:
    return "\t".join(
        [
            label,
            _fmt_num(summary.mean),
            _fmt_num(summary.median),
            _fmt_num(summary.ci_lo),
            _fmt_num(summary.ci_hi),
            _fmt_p(summary.perm_pvalue),
            _fmt_p(summary.wilcoxon_pvalue),
        ]
    )


def write_tsv(rows: list[tuple[str, PairedDiffSummary]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [TSV_HEADER]
    for label, summary in rows:
        lines.append(_tsv_row(label, summary))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floor", type=float, default=2.0)
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument("--model", default=None, help="Display name; default = all complete models")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"TSV path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--no-tsv",
        action="store_true",
        help="Print only; do not write a TSV",
    )
    args = ap.parse_args()

    override = load_deltae_override(resolve_deltae_json(args.baseline))
    seed_dirs = geom_seed_paths()
    seed_metas = {
        d: augment_meta(
            load_metadata_index(d / "prompt_metadata.jsonl"),
            override,
            strict=override is not None,
        )
        for d in seed_dirs
    }
    molecules = sorted(set.intersection(*(set(m) for m in seed_metas.values())))
    meta0 = seed_metas[seed_dirs[0]]
    uff = _uff_per_molecule(meta0, geom_conformers_dir(), args.floor)
    print(f"UFF: {len(uff)}/{len(molecules)} molecules @ floor {args.floor}")

    if args.model is not None:
        files = {args.model: model_files_by_display("geom_benchmark")[args.model]}
    else:
        files = dict(complete_geom_models(deltae_override=override, warn_unregistered=False))

    print(
        f"{'model':40s} {'n':>3s} {'mean_d':>8s} {'median':>8s} "
        f"{'CI_lo':>8s} {'CI_hi':>8s} {'t_p':>8s} {'perm_p':>8s} {'W_p':>8s}"
    )
    rows: list[tuple[str, PairedDiffSummary]] = []
    for display, fname in files.items():
        llm = _llm_mean_per_molecule(fname, seed_metas, molecules, args.floor)
        common = sorted(set(llm) & set(uff))
        if not common:
            print(f"{display:40s} skip (no overlap)")
            continue
        summary = summarize_llm_vs_baseline(
            [llm[m] for m in common],
            [uff[m] for m in common],
            n_perm=args.n_perm,
            seed=args.seed,
        )
        label = short_label_for_display(display)
        rows.append((label, summary))
        print(
            f"{display:40s} {summary.n:3d} {summary.mean:+8.3f} {summary.median:+8.3f} "
            f"{summary.ci_lo:+8.3f} {summary.ci_hi:+8.3f} "
            f"{summary.t_pvalue:8.3g} {summary.perm_pvalue:8.3g} "
            f"{summary.wilcoxon_pvalue:8.3g}"
        )

    rows.sort(key=lambda r: -r[1].mean)
    if not args.no_tsv:
        out = args.out
        if args.floor != 2.0 and out == DEFAULT_OUT:
            tag = f"{args.floor:g}".replace(".", "")
            out = out.with_name(f"paired_llm_minus_uff_floor{tag}.tsv")
        write_tsv(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
