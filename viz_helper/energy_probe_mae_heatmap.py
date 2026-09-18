"""Heatmap of energy-probe absolute errors (MAE in cells).

Reads geom-qm9/energy_probe_*.jsonl — one conf per molecule per model.
Cell = |pred − truth| kcal/mol; column mean printed as MAE.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import models_by_id, short_label_for_display
from viz.plot_style import DOUBLE_COLUMN_WIDTH, FONTSIZE, FONTSIZE_LABEL, plt, style_ax

DEFAULT_PROBE_DIR = Path("geom-qm9")
DEFAULT_MOL_TABLE = Path("figures/r2scan3c/seed_spread_heatmap_molecules.tsv")
DEFAULT_TEXT_DIR = Path("texts/seed_avg")
DEFAULT_OUT = Path("figures/r2scan3c/energy_probe_mae_heatmap.png")


MAE_CMAP_STOPS = (
    (0.0, "#2e7d32"),
    (0.25, "#4fc3f7"),
    (0.5, "#ffffff"),
    (0.75, "#ef6c00"),
    (1.0, "#c62828"),
)


def mae_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("mae_gwor", MAE_CMAP_STOPS)


def load_molecule_order(table: Path) -> list[tuple[str, str]]:
    """Return [(M01, molecule_id), ...] from seed-spread TSV."""
    rows: list[tuple[str, str]] = []
    for line in table.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        label, mol_id, *_ = line.split("\t")
        rows.append((label, mol_id))
    return rows


def load_probe_matrix(
    probe_dir: Path,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """models (unsorted), lookup[model][molecule] = abs_error_kcal."""
    lookup: dict[str, dict[str, float]] = {}
    for path in sorted(probe_dir.glob("energy_probe_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row["model"])
            mol = str(row["molecule"])
            err = row.get("abs_error_kcal")
            if err is None or row.get("api_error"):
                continue
            lookup.setdefault(model, {})[mol] = float(err)
    models = list(lookup)
    return models, lookup


def _display_label(model: str) -> str:

    base = model.rsplit("/", 1)[-1]
    short = short_label_for_display(base)
    return short if short != base else base


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_tau_rank(text_dir: Path = DEFAULT_TEXT_DIR) -> dict[str, int]:
    """model_id -> rank by mean seed-averaged τ (0 = best), the heatmap order."""
    tau: dict[str, float] = {}
    if text_dir.is_dir():
        for f in text_dir.glob("*.jsonl"):
            vals = [
                json.loads(line)["tau"]
                for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("tau") is not None
            ]
            if vals:
                tau[f.stem] = float(np.mean(vals))
    return {mid: i for i, mid in enumerate(sorted(tau, key=lambda k: -tau[k]))}


def _probe_model_id(model: str, specs: dict) -> str | None:
    base = _norm(model.rsplit("/", 1)[-1])
    fuzzy = None
    for mid, spec in specs.items():
        for cand in (_norm(spec.short or ""), _norm(spec.display or "")):
            if not cand:
                continue
            if cand == base:
                return mid
            if cand.startswith(base) or base.startswith(cand):
                fuzzy = fuzzy or mid
    return fuzzy


def plot_mae_heatmap(
    *,
    probe_dir: Path = DEFAULT_PROBE_DIR,
    molecule_table: Path = DEFAULT_MOL_TABLE,
    out: Path = DEFAULT_OUT,
    vmax: float | None = None,
) -> Path:
    mol_rows = load_molecule_order(molecule_table)
    models, lookup = load_probe_matrix(probe_dir)
    if not models:
        raise SystemExit(f"No energy_probe_*.jsonl under {probe_dir}")


    specs = models_by_id()
    tau_rank = load_tau_rank()
    unknown_rank = len(tau_rank) + 1

    def col_mae(m: str) -> float:
        vals = [lookup[m][mol] for _, mol in mol_rows if mol in lookup[m]]
        return float(np.mean(vals)) if vals else float("inf")

    def order_key(m: str) -> tuple[int, float]:
        mid = _probe_model_id(m, specs)
        return (tau_rank.get(mid, unknown_rank) if mid else unknown_rank, col_mae(m))

    col_models = sorted(models, key=order_key)
    n_cols = len(col_models)
    n_rows = len(mol_rows)

    matrix = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, (_, mol) in enumerate(mol_rows):
        for j, model in enumerate(col_models):
            if mol in lookup[model]:
                matrix[i, j] = lookup[model][mol]

    finite = matrix[np.isfinite(matrix)]
    if vmax is None:
        vmax = float(np.percentile(finite, 95)) if finite.size else 5.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = mae_cmap()

    col_w = 0.42
    fig_w = max(DOUBLE_COLUMN_WIDTH, col_w * n_cols + 1.6)
    fig_h = max(5.5, 0.20 * n_rows + 1.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, right = 0.10, 0.92
    bottom, top = 0.18, 0.98
    ax = fig.add_axes([left, bottom, right - left, top - bottom])

    x_min, x_max = -0.5, n_cols - 0.5
    y_top, y_bottom = -0.5, n_rows - 0.5
    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        extent=[x_min, x_max, y_bottom, y_top],
        origin="upper",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue

            t = min(1.0, max(0.0, val / vmax))
            color = "#111111" if 0.25 < t < 0.75 else "#ffffff"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )

    mae_y = -0.85
    for j, model in enumerate(col_models):
        mae = col_mae(model)
        ax.text(
            j,
            mae_y,
            f"{mae:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="#111111",
        )
    ax.text(
        -0.85,
        mae_y,
        "MAE",
        ha="right",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="#333333",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, mae_y - 0.25)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([lab for lab, _ in mol_rows], fontsize=FONTSIZE)
    ax.set_ylabel("Molecule (seed-spread order)", fontsize=FONTSIZE_LABEL)

    tick_labels = [_display_label(m) for m in col_models]
    ax.set_xticks(np.arange(n_cols, dtype=float))
    ax.set_xticklabels(
        tick_labels,
        fontsize=6,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=2.5, width=0.6, pad=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.92)
    cbar.set_label(r"$|E_{\mathrm{pred}}-E_{\mathrm{GEOM-QM9}}|$ (kcal/mol)", fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 1)

    style_ax(ax, square=False)
    ax.spines["top"].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=2000, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved {out}")
    for model in col_models:
        print(f"  {_display_label(model):20s}  MAE={col_mae(model):.3f}  n={sum(1 for _, m in mol_rows if m in lookup[model])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    ap.add_argument("--molecule-table", type=Path, default=DEFAULT_MOL_TABLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vmax", type=float, default=None, help="Color scale max (default: 95th pct)")
    args = ap.parse_args()
    plot_mae_heatmap(
        probe_dir=args.probe_dir,
        molecule_table=args.molecule_table,
        out=args.out,
        vmax=args.vmax,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
