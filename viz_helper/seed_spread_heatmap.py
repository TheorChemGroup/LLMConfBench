"""Molecule × model heatmap + Min@10 counts (replaces seed_spread strip plots).

Rows = 27 molecules sorted by mean LLM CRA on the equal-H-bond (2,997) pairs,
best first. Columns = all models (LLMs + GFN-FF + UFF + MMFF94) sorted
left-to-right by mean CRA on the same pairs. Cells show CRA on those pairs. Coloured Min@10 triplets (0/3, 1–2/3, 3/3) sit above
the heatmap; model names below.

  python viz_helper/seed_spread_heatmap.py
  python viz_helper/seed_spread_heatmap.py --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import (
    geom_conformers_dir,
    geom_seed_paths,
    model_files_by_display,
    resolve_deltae_json,
    short_label_for_display,
)
from metrics.conformer_min_in_topk import letters_at_global_minimum, min_in_topk
from metrics.grouped_metrics import mean_std_ci
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)
from metrics.option2_metrics import load_valid_answers
from viz.cohort import complete_geom_models
from viz.plot_style import DOUBLE_COLUMN_WIDTH, FONTSIZE, FONTSIZE_LABEL, plt, style_ax
from viz_helper.decoy_common import (
    bench_molecules,
    equal_hbond_pair_tables,
    hb_count_map,
    load_r2scan,
)
from viz_helper.plot_metrics_vs_release import (
    MMFF94_NAME,
    _smiles_by_folder,
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
)

MIN10_COLOR_0 = "#111111"
MIN10_COLOR_12 = "#5c6bc0"
MIN10_COLOR_3 = "#7b1fa2"
COUNTS_FONTSIZE = 8
COUNTS_XOFF_PT = 6.0
COUNTS_YOFF_PT = 4.0

FF_MODELS = ("GFN-FF", "UFF", MMFF94_NAME)
DEFAULT_OUT = Path("figures/seed_spread_heatmap.png")
DEFAULT_MOLECULE_TABLE = Path("figures/seed_spread_heatmap_molecules.tsv")
TAU_CMAP_VMIN = -1.0
TAU_CMAP_VMAX = 1.0
TAU_CMAP_STOPS = (
    (TAU_CMAP_VMIN, "#c62828"),
    (-0.5, "#ef6c00"),
    (0.0, "#ffffff"),
    (0.5, "#4fc3f7"),
    (TAU_CMAP_VMAX, "#2e7d32"),
)

TOP_K = 10
SEED_DIRS = geom_seed_paths()
CONFORMERS_DIR = geom_conformers_dir()


def tau_ranking_cmap() -> LinearSegmentedColormap:
    span = TAU_CMAP_VMAX - TAU_CMAP_VMIN
    stops = [((value - TAU_CMAP_VMIN) / span, color) for value, color in TAU_CMAP_STOPS]
    return LinearSegmentedColormap.from_list("tau_rowgb", stops)


def _tau_on_equal_hb(
    pairs: list[tuple[int, int]],
    public_map: dict,
    ranking: list[str],
) -> float | None:
    """CRA on the equal-H-bond pairs (2,997 pairs) for one ranking."""
    conf_to_public = {int(index): str(label) for label, index in public_map.items()}
    pos = {label: i for i, label in enumerate(ranking)}
    ok = tot = 0
    for lo, hi in pairs:
        llo = conf_to_public.get(lo)
        lhi = conf_to_public.get(hi)
        if llo is None or lhi is None or llo not in pos or lhi not in pos:
            continue
        tot += 1
        ok += pos[llo] < pos[lhi]
    return (2.0 * ok / tot - 1.0) if tot else None


def _fmt_mean_ci(means) -> str:
    arr = np.asarray(means, dtype=float)
    mean, _std, _se, lo, hi = mean_std_ci(arr)
    half = 0.5 * (hi - lo)
    return f"mean_tau={mean:+.3f} ± {half:.3f}"


def _forcefield_column(meta, molecules, ranking_getter, pair_table):
    """One column: (mol, tau, tau, tau, n_hits). n_hits is 3 if min@10 else 0."""
    per_mol = []
    for mol in molecules:
        folder = CONFORMERS_DIR / mol
        mol_meta = meta.get(mol, {})
        public_map = mol_meta.get("public_label_to_conf_index", {})
        letter_to_de = mol_meta.get("letter_to_deltaE_kcal_mol", {})
        if not folder.is_dir() or not public_map or not letter_to_de:
            continue
        conf_to_public = {int(index): str(label) for label, index in public_map.items()}
        rank = ranking_getter(folder)
        if rank is None:
            continue
        prediction = [conf_to_public[index] for index in rank if index in conf_to_public]
        if len(prediction) != 30:
            continue
        tau = _tau_on_equal_hb(pair_table.get(mol, []), public_map, prediction)
        if tau is None:
            continue
        hit = min_in_topk(
            prediction, letters_at_global_minimum(letter_to_de), TOP_K,
        )
        per_mol.append((mol, tau, tau, tau, 3 if hit else 0))
    per_mol.sort(key=lambda row: row[1])
    return per_mol


def load_data(
    deltae_override: dict | None = None,
    model_files: dict[str, str] | None = None,
):
    seed_metas = {}
    for seed_dir in SEED_DIRS:
        meta_path = seed_dir / "prompt_metadata.jsonl"
        if meta_path.is_file():
            seed_metas[seed_dir] = augment_meta(
                load_metadata_index(meta_path),
                deltae_override,
                strict=deltae_override is not None,
            )
    if len(seed_metas) != len(SEED_DIRS):
        raise ValueError(
            "seed_spread_heatmap requires all configured seed metadata"
        )
    molecules = sorted(
        set.intersection(*(set(meta) for meta in seed_metas.values()))
    )
    print(f"Loaded {len(molecules)} molecules from 3 seeds")
    r2scan = load_r2scan()
    pair_table = equal_hbond_pair_tables(
        r2scan, hb_count_map(r2scan, bench_molecules())
    )
    print(
        "Metric: CRA on equal-H-bond pairs "
        f"({sum(len(v) for v in pair_table.values())} pairs)"
    )

    llm_data = {}
    specs = (
        list(model_files.items())
        if model_files is not None
        else list(model_files_by_display("geom_benchmark").items())
    )
    for model_name, fname in specs:
        mol_taus = {mol: [] for mol in molecules}
        mol_hits = {mol: 0 for mol in molecules}
        for seed_dir in SEED_DIRS:
            if seed_dir not in seed_metas:
                continue
            meta = seed_metas[seed_dir]
            answers = load_valid_answers(seed_dir / fname, meta)
            if not answers:
                continue
            for mol in molecules:
                if mol not in answers:
                    continue
                letter_to_de = meta[mol]["letter_to_deltaE_kcal_mol"]
                tau = _tau_on_equal_hb(
                    pair_table.get(mol, []),
                    meta[mol]["public_label_to_conf_index"],
                    answers[mol],
                )
                if tau is not None:
                    mol_taus[mol].append(tau)
                min_letters = letters_at_global_minimum(letter_to_de)
                if min_in_topk(answers[mol], min_letters, TOP_K):
                    mol_hits[mol] += 1

        per_mol = []
        for mol in molecules:
            taus = mol_taus[mol]
            if len(taus) < 3:
                continue
            per_mol.append((mol, np.mean(taus), np.min(taus), np.max(taus), mol_hits[mol]))
        per_mol.sort(key=lambda d: d[1])
        if not per_mol:
            continue
        means = [d[1] for d in per_mol]
        n3 = sum(1 for d in per_mol if d[4] == 3)
        n12 = sum(1 for d in per_mol if d[4] in (1, 2))
        n0 = sum(1 for d in per_mol if d[4] == 0)
        print(
            f"  {model_name}: {_fmt_mean_ci(means)}, "
            f"min@{TOP_K} 3/3={n3} 1–2/3={n12} 0/3={n0}"
        )
        llm_data[model_name] = per_mol

    meta42 = seed_metas[SEED_DIRS[0]]
    print("Computing UFF, GFN-FF and MMFF94 rankings...")
    for ff_name, getter in (
        ("UFF", get_uff_ranking),
        ("GFN-FF", get_gfnff_ranking),
        (MMFF94_NAME, get_mmff94_ranking),
    ):
        col = _forcefield_column(meta42, molecules, getter, pair_table)
        if not col:
            print(f"  {ff_name}: no molecules")
            continue
        means = [row[1] for row in col]
        n_hit = sum(1 for row in col if row[4] >= 3)
        n_miss = len(col) - n_hit
        print(
            f"  {ff_name}: {_fmt_mean_ci(means)}, "
            f"min@{TOP_K} hit={n_hit} miss={n_miss}"
        )
        llm_data[ff_name] = col

    return molecules, llm_data


def _read_order(path: Path) -> list[str] | None:
    """Molecule order from an existing label table (labels 1…27)."""
    if not path.is_file():
        return None
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) > 1 and parts[1]:
            order.append(parts[1])
    return order or None


def heatmap_row_order(
    llm_data: dict[str, list[tuple]],
    fixed_order: list[str] | None = None,
) -> list[str]:
    present = {mol for model in llm_data for mol, *_ in llm_data[model]}
    if fixed_order:
        ordered = [mol for mol in fixed_order if mol in present]
        return ordered + sorted(present - set(ordered))
    llm_models = [name for name in llm_data if name not in FF_MODELS]
    mol_mean_llm: dict[str, list[float]] = {}
    for model in llm_models:
        for mol, mean_tau, *_ in llm_data[model]:
            mol_mean_llm.setdefault(mol, []).append(mean_tau)
    return sorted(
        mol_mean_llm,
        key=lambda mol: float(np.mean(mol_mean_llm[mol])),
        reverse=True,
    )


def mean_llm_tau_by_molecule(llm_data: dict[str, list[tuple]]) -> dict[str, float]:
    llm_models = [name for name in llm_data if name not in FF_MODELS]
    mol_mean_llm: dict[str, list[float]] = {}
    for model in llm_models:
        for mol, mean_tau, *_ in llm_data[model]:
            mol_mean_llm.setdefault(mol, []).append(mean_tau)
    return {mol: float(np.mean(vals)) for mol, vals in mol_mean_llm.items()}


def write_molecule_label_table(
    llm_data: dict[str, list[tuple]],
    out: Path,
    row_order: list[str] | None = None,
) -> Path:
    """Ranking row labels for seed_spread_heatmap (best molecule = 1),
    including the two pilot molecules (mPM/mPP) ranked inline."""
    row_mols = heatmap_row_order(
        llm_data, fixed_order=row_order or _read_order(out)
    )
    smiles_map = _smiles_by_folder()
    mean_tau = mean_llm_tau_by_molecule(llm_data)
    lines = ["label\tmolecule_id\tsmiles\tmean_llm_tau"]
    for i, mol in enumerate(row_mols, start=1):
        smiles = smiles_map.get(mol, "")
        lines.append(f"{i}\t{mol}\t{smiles}\t{mean_tau.get(mol, float('nan')):.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}")
    return out


def _tau_matrix(
    llm_data: dict[str, list[tuple]],
    molecules: list[str],
    col_models: list[str],
) -> np.ndarray:
    lookup: dict[str, dict[str, float]] = {}
    for model, per_mol in llm_data.items():
        lookup[model] = {row[0]: float(row[1]) for row in per_mol}
    return np.array(
        [[lookup.get(model, {}).get(mol, np.nan) for model in col_models] for mol in molecules],
        dtype=float,
    )


def _min10_counts(per_mol: list[tuple]) -> tuple[int, int, int]:
    n3 = sum(1 for row in per_mol if row[4] >= 3)
    n12 = sum(1 for row in per_mol if row[4] in (1, 2))
    n0 = sum(1 for row in per_mol if row[4] == 0)
    return n0, n12, n3


def _draw_color_counts_cell(ax, x: float, y: float, n3: int, n12: int, n0: int) -> None:
    """Counts "n0,n12,n3" drawn as one 45°-rotated string (colour per category)."""
    import math

    fontsize = COUNTS_FONTSIZE
    parts = [
        (str(n0), MIN10_COLOR_0),
        (",", "#333333"),
        (str(n12), MIN10_COLOR_12),
        (",", "#333333"),
        (str(n3), MIN10_COLOR_3),
    ]
    char_w = fontsize * 0.62
    width = sum(len(text) * char_w for text, _ in parts)
    angle = math.radians(45)
    cursor = -width / 2.0
    for text, color in parts:
        w = len(text) * char_w
        shift = (cursor + w / 2.0)
        ax.annotate(
            text,
            (x, y),
            xytext=(COUNTS_XOFF_PT + shift * math.cos(angle),
                    COUNTS_YOFF_PT + shift * math.sin(angle)),
            textcoords="offset points",
            rotation=45,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            annotation_clip=False,
        )
        cursor += w


def plot_heatmap_panels(
    llm_data: dict[str, list[tuple]],
    out: Path,
    *,
    row_order: list[str] | None = None,
) -> None:
    col_models = sorted(
        llm_data,
        key=lambda name: float(np.mean([row[1] for row in llm_data[name]])),
        reverse=True,
    )
    n_cols = len(col_models)

    row_mols = heatmap_row_order(llm_data, fixed_order=row_order)
    n_rows = len(row_mols)
    row_labels = [f"{i + 1}" for i in range(n_rows)]

    matrix = _tau_matrix(llm_data, row_mols, col_models)
    norm = Normalize(vmin=TAU_CMAP_VMIN, vmax=TAU_CMAP_VMAX)
    cmap = tau_ranking_cmap()

    col_w = 0.28 if n_cols > 20 else 0.34
    fig_w = max(DOUBLE_COLUMN_WIDTH, col_w * n_cols + 1.8)
    fig_h = max(5.5, 0.17 * n_rows + 2.0)
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right = 0.10, 0.94
    bottom = 0.16
    top = 0.98
    ax_hm = fig.add_axes([left, bottom, right - left, top - bottom])

    x_centers = np.arange(n_cols, dtype=float)
    x_min, x_max = -0.5, n_cols - 0.5
    y_top = -0.5
    counts_y = -1.32
    y_margin_top = counts_y - 0.30
    y_bottom = n_rows - 0.5

    im = ax_hm.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        extent=[x_min, x_max, y_bottom, y_top],
        origin="upper",
    )
    ax_hm.set_xlim(x_min, x_max)
    ax_hm.set_ylim(y_bottom, y_margin_top)
    ax_hm.set_yticks(range(n_rows))
    ax_hm.set_yticklabels(row_labels, fontsize=FONTSIZE)
    ax_hm.set_ylabel("Molecule (ordered by mean LLM accuracy)", fontsize=FONTSIZE_LABEL)

    for j, model in enumerate(col_models):
        n0, n12, n3 = _min10_counts(llm_data[model])
        _draw_color_counts_cell(ax_hm, float(j), counts_y, n3, n12, n0)

    tick_labels = [
        name if name in FF_MODELS else short_label_for_display(name)
        for name in col_models
    ]
    tick_fs = 5 if n_cols > 20 else 6
    ax_hm.set_xticks(x_centers)
    ax_hm.set_xticklabels(
        tick_labels,
        fontsize=tick_fs,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    for label, name in zip(ax_hm.get_xticklabels(), col_models):
        if name in FF_MODELS:
            label.set_fontweight("bold")
    ax_hm.xaxis.set_ticks_position("bottom")
    ax_hm.tick_params(
        axis="x",
        bottom=True,
        top=False,
        labelbottom=True,
        length=2.5,
        width=0.6,
        direction="out",
        pad=1,
        labelsize=tick_fs,
    )

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02, shrink=0.92)
    cbar.set_label("CRA across fair pairs", fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 1)

    style_ax(ax_hm, square=False)
    ax_hm.spines["top"].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"\nSaved to {out}")


def collect_and_write(
    *,
    baseline: str = "r2scan3c",
    model_files: dict[str, str] | None = None,
    deltae_json: Path | None = None,
    out: Path | None = None,
    molecule_table: Path | None = None,
) -> Path:
    out = out or DEFAULT_OUT
    molecule_table = molecule_table or DEFAULT_MOLECULE_TABLE
    override = load_deltae_override(resolve_deltae_json(baseline, deltae_json))
    if model_files is None:
        model_files = dict(
            complete_geom_models(deltae_override=override, warn_unregistered=False)
        )
    _molecules, llm_data = load_data(override, model_files)
    row_order = heatmap_row_order(llm_data)
    plot_heatmap_panels(llm_data, out, row_order=row_order)
    write_molecule_label_table(llm_data, molecule_table, row_order=row_order)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--molecule-table",
        type=Path,
        default=None,
        help="TSV mapping 1…27 to folder id and SMILES (heatmap row order)",
    )
    ap.add_argument("--baseline", default="r2scan3c")
    ap.add_argument("--model-files", type=str, default=None)
    ap.add_argument("--deltae-json", type=Path, default=None)
    args = ap.parse_args()
    collect_and_write(
        baseline=args.baseline,
        model_files=json.loads(args.model_files) if args.model_files else None,
        deltae_json=args.deltae_json,
        out=args.out,
        molecule_table=args.molecule_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
