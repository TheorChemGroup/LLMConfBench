#!/usr/bin/env python3

"""Table: reliable-pair Kendall τ @ |ΔE| floor 2.0 kcal/mol.

Per model, over molecules (3 seeds averaged first): mean, median, sample SD.

  python viz_helper/tau_floor2_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import (
    TAU_FLOOR,
    geom_conformers_dir,
    geom_seed_paths,
    resolve_deltae_json,
    short_label_for_display,
)
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import load_valid_answers, score_molecule
from viz.cohort import complete_geom_models
from viz.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    FONTSIZE,
    FONTSIZE_LABEL,
    plt,
    save_fig,
)
from viz_helper.plot_metrics_vs_release import get_gfnff_ranking, get_uff_ranking
from viz_helper.seed_spread_heatmap import MMFF94_NAME, get_mmff94_ranking

FLOOR = TAU_FLOOR
TAU_KEY = f"tau_f{FLOOR}"
CONFORMERS_DIR = geom_conformers_dir()
DEFAULT_OUT = Path("figures/r2scan3c/tau_floor2_table.png")
FORCEFIELDS = {"UFF", "GFN-FF", MMFF94_NAME}
COL_LABELS = ["Model", "mean τ", "median τ", "SD"]
COL_WIDTHS = [0.42, 0.20, 0.22, 0.16]


def _row_from_vals(name: str, vals: np.ndarray) -> tuple[str, float, float, float, int]:
    return (
        name,
        float(np.mean(vals)),
        float(np.median(vals)),
        float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        int(vals.size),
    )


def _tau_for_ranking(letter_to_de: dict, ranking: list[str]) -> float | None:
    scored = score_molecule(ranking, letter_to_de, [FLOOR])
    if not scored.get("valid"):
        return None
    value = scored.get(TAU_KEY)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def forcefield_rows(
    molecules: list[str],
    meta: dict,
) -> list[tuple[str, float, float, float, int]]:
    """One ranking per molecule (no seeds): reliable τ @ 2.0."""
    out: list[tuple[str, float, float, float, int]] = []
    for name, getter in (
        ("UFF", get_uff_ranking),
        ("GFN-FF", get_gfnff_ranking),
        (MMFF94_NAME, get_mmff94_ranking),
    ):
        vals: list[float] = []
        for mol in molecules:
            folder = CONFORMERS_DIR / mol
            mol_meta = meta.get(mol, {})
            pub_map = mol_meta.get("public_label_to_conf_index", {})
            letter_de = mol_meta.get("letter_to_deltaE_kcal_mol", {})
            conf_to_pub = {int(v): str(k) for k, v in pub_map.items()}
            rank_idx = getter(folder) if folder.is_dir() else None
            if not rank_idx or not letter_de:
                continue
            pred = [conf_to_pub[idx] for idx in rank_idx if idx in conf_to_pub]
            tau = _tau_for_ranking(letter_de, pred)
            if tau is not None:
                vals.append(tau)
        if vals:
            arr = np.asarray(vals, dtype=float)
            print(f"  {name}: {len(arr)}/{len(molecules)} molecules, mean τ@2.0={arr.mean():+.3f}")
            out.append(_row_from_vals(name, arr))
        else:
            print(f"  {name}: no scores")
    return out


def model_rows(
    seed_metas: dict[Path, dict],
    molecules: list[str],
    model_files: dict[str, str],
    extra: list[tuple[str, float, float, float, int]] = (),
) -> list[tuple[str, float, float, float, int]]:
    seed_dirs = geom_seed_paths()
    rows = list(extra)
    for display, filename in model_files.items():
        per_seed_answers = [
            load_valid_answers(seed_dir / filename, seed_metas[seed_dir])
            for seed_dir in seed_dirs
        ]
        mol_means: list[float] = []
        for mol in molecules:
            seed_vals: list[float] = []
            for seed_dir, answers in zip(seed_dirs, per_seed_answers, strict=True):
                if mol not in answers:
                    continue
                meta = seed_metas[seed_dir]
                tau = _tau_for_ranking(meta[mol]["letter_to_deltaE_kcal_mol"], answers[mol])
                if tau is not None:
                    seed_vals.append(tau)
            if len(seed_vals) == len(seed_dirs):
                mol_means.append(float(np.mean(seed_vals)))
        if not mol_means:
            continue
        arr = np.asarray(mol_means, dtype=float)
        label = short_label_for_display(display)
        rows.append(_row_from_vals(label, arr))
    rows.sort(key=lambda r: -r[1])
    return rows


def write_tsv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["model\tmean_tau_f2.0\tmedian_tau_f2.0\tsd_tau_f2.0\tn_molecules"]
    for name, mean, median, sd, n in rows:
        lines.append(f"{name}\t{_fmt_tau_plain(mean)}\t{_fmt_tau_plain(median)}\t{sd:.3f}\t{n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def _fmt_tau(value: float) -> str:
    text = f"{value:+.3f}"
    if text in {"+0.000", "-0.000"}:
        return "0.000"
    return text


def _fmt_tau_plain(value: float) -> str:
    text = f"{value:.3f}"
    if text in {"0.000", "-0.000"}:
        return "0.000"
    return text


def _body_cells(
    rows: list[tuple[str, float, float, float, int]],
    n_side: int,
) -> list[list[str]]:
    cells = [
        [name, _fmt_tau(mean), _fmt_tau(median), f"{sd:.3f}"]
        for name, mean, median, sd, _n in rows
    ]
    while len(cells) < n_side:
        cells.append(["", "", "", ""])
    return cells


def _style_table(table, rows: list[tuple[str, float, float, float, int]]) -> None:
    italic = {i + 1 for i, (name, *_rest) in enumerate(rows) if name in FORCEFIELDS}
    table.auto_set_font_size(False)
    table.set_fontsize(FONTSIZE)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.4)
        if row_index == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", fontsize=FONTSIZE)
        elif row_index % 2 == 0:
            cell.set_facecolor("#fafafa")
        if row_index in italic:
            cell.set_text_props(style="italic")
        if column_index == 0:
            cell.get_text().set_ha("left")


def plot_table(rows, out: Path, *, n_mol: int) -> None:
    n_side = max(1, (len(rows) + 1) // 2)
    left_rows = rows[:n_side]
    right_rows = rows[n_side:]
    title_in = 0.30
    row_in = 0.20
    fig_h = title_in + row_in * (n_side + 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH * 1.55, fig_h),
        gridspec_kw={"wspace": 0.08},
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        top=1.0 - title_in / fig_h,
        bottom=0.02,
        wspace=0.08,
    )
    for ax, slice_rows in zip(axes, (left_rows, right_rows), strict=True):
        ax.set_axis_off()
        table = ax.table(
            cellText=_body_cells(slice_rows, n_side),
            colLabels=COL_LABELS,
            loc="upper center",
            cellLoc="center",
            colWidths=COL_WIDTHS,
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        _style_table(table, slice_rows)
    fig.text(
        0.5,
        1.0 - 0.05 / fig_h,
        f"Conformers Ranking Accuracy (|ΔE| ≥ 2.0 kcal/mol)  (n = {n_mol} molecules)",
        ha="center",
        va="top",
        fontsize=FONTSIZE_LABEL,
        transform=fig.transFigure,
    )
    save_fig(fig, out)


def collect_and_write(
    *,
    baseline: str = "r2scan3c",
    deltae_json: Path | None = None,
    out: Path | None = None,
) -> Path:
    if out is None:
        out = DEFAULT_OUT
    override = load_deltae_override(resolve_deltae_json(baseline, deltae_json))
    model_files = dict(
        complete_geom_models(deltae_override=override, warn_unregistered=False)
    )
    if not model_files:
        raise SystemExit(f"No models with 3 complete seeds for {baseline}.")

    seed_dirs = geom_seed_paths()
    seed_metas: dict[Path, dict] = {}
    for seed_dir in seed_dirs:
        seed_metas[seed_dir] = augment_meta(
            load_metadata_index(seed_dir / "prompt_metadata.jsonl"),
            override,
            strict=override is not None,
        )
    molecules = sorted(set.intersection(*(set(meta) for meta in seed_metas.values())))
    print(f"Loaded {len(molecules)} molecules from 3 seeds")

    meta42 = seed_metas[seed_dirs[0]]
    print("Computing UFF, GFN-FF and MMFF94 rankings...")
    rows = model_rows(
        seed_metas,
        molecules,
        model_files,
        extra=forcefield_rows(molecules, meta42),
    )
    write_tsv(rows, out.with_suffix(".tsv"))
    plot_table(rows, out, n_mol=len(molecules))
    for name, mean, median, sd, n in rows:
        print(f"{name:20s}  mean={_fmt_tau(mean)}  median={_fmt_tau(median)}  sd={sd:.3f}  n={n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument("--deltae-json", type=Path, default=None)
    args = ap.parse_args()
    collect_and_write(
        baseline=args.baseline,
        deltae_json=args.deltae_json,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
