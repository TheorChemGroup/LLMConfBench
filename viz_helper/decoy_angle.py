"""Decoy-pair accuracy versus H-bond geometry (Fig. 3c).

For every unconventional (decoy) pair, the strongest H-bond of the more
H-bonded conformer is located and its H...O/N distance and donor angle are
compared with the mean LLM accuracy on that pair.

    python viz_helper/decoy_angle.py
"""
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(REPO / "viz"), str(Path(__file__).parent)]

from config.models_config import geom_conformers_dir  # noqa: E402
from decoy_common import bonded_graph, read_xyz  # noqa: E402
from viz.plot_style import (  # noqa: E402
    FONTSIZE,
    FONTSIZE_LABEL,
    ONE_HALF_COLUMN_WIDTH,
    SAVE_DPI,
    TEXT_COLOR,
    plt,
    save_fig,
    style_ax,
)

KEYS = ["molecule_number", "conf1", "conf2"]
COLORS = {8: "#1565c0", 16: "#c62828", 21: "#ef6c00", 22: "#6a1b9a",
          4: "#2e7d32"}
LABELS = {8: "mol 8", 16: "mol 16", 21: "mol 21", 22: "mol 22", 4: "mol 4"}
GREY = "#b0b0b0"


def contacts(atoms, bonded, h_max=2.5, ang_min=120.0):
    """Geometric donor-H...acceptor contacts as (r, angle)."""
    xyz = {i: a[1:] for i, a in enumerate(atoms)}
    out = []
    for i, a in enumerate(atoms):
        if a[0] != "H":
            continue
        donors = [j for j in bonded.get(i, []) if atoms[j][0] in ("O", "N")]
        if len(donors) != 1:
            continue
        d = donors[0]
        hd = np.array(xyz[d]) - np.array(xyz[i])
        nd = float(np.linalg.norm(hd))
        for k, b in enumerate(atoms):
            if b[0] not in ("O", "N") or k in (i, d):
                continue
            ha = np.array(xyz[k]) - np.array(xyz[i])
            r = float(np.linalg.norm(ha))
            if r > h_max or r < 1e-9:
                continue
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, float(np.dot(hd, ha)) / (nd * r)))))
            if ang >= ang_min:
                out.append((r, ang))
    return out


def geometry_table() -> pd.DataFrame:
    labels = pd.read_csv(OUT / "seed_spread_heatmap_molecules.tsv", sep="\t")
    mol_id = dict(zip(labels.label, labels.molecule_id))
    conf = geom_conformers_dir()

    d = pd.read_csv(REPO / "pair_runs_reliable.csv")
    u = d.drop_duplicates(KEYS).copy()
    u["cls"] = np.where(
        u.hb1 == u.hb2, "eq",
        np.where(((u.hb1 > u.hb2) & (u.dE_signed > 0))
                 | ((u.hb2 > u.hb1) & (u.dE_signed < 0)), "fav", "dec"))
    u["rich"] = np.where(u.hb1 >= u.hb2, u.conf1, u.conf2)

    rows = []
    for r in u[u.cls == "dec"].itertuples():
        mid = mol_id[r.molecule_number]
        path = conf / mid / f"{mid}_conf_{r.rich}.xyz"
        if not path.is_file():
            continue
        atoms = read_xyz(path)
        cs = contacts(atoms, bonded_graph(atoms))
        if not cs:
            continue
        best = min(cs, key=lambda c: (c[0], -c[1]))
        rows.append({"molecule": r.molecule_number, "conf1": r.conf1,
                     "conf2": r.conf2, "rich": r.rich,
                     "dE": abs(r.dE_signed), "r": best[0], "angle": best[1]})
    g = pd.DataFrame(rows)
    acc = (d.groupby(KEYS).CRA.mean().rename("accuracy").reset_index()
           .rename(columns={"molecule_number": "molecule"}))
    return g.merge(acc, on=["molecule", "conf1", "conf2"])


def plot(M: pd.DataFrame, out: Path) -> Path:
    x, y, mols = M.angle.values, M.accuracy.values, M.molecule.values
    fig = plt.figure(figsize=(ONE_HALF_COLUMN_WIDTH, ONE_HALF_COLUMN_WIDTH))
    ax = fig.add_axes([0.16, 0.16, 0.80, 0.78])

    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs),
            color=TEXT_COLOR, lw=1.1, zorder=2)
    rng = np.random.default_rng(0)
    uniq = np.unique(mols)
    boot = []
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sub = pd.concat([M[M.molecule == m] for m in pick])
        boot.append(np.polyval(np.polyfit(sub.angle.values,
                                          sub.accuracy.values, 1), xs))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=TEXT_COLOR, alpha=0.10, lw=0)

    for mol in uniq:
        q = M[M.molecule == mol]
        if mol in COLORS:
            ax.scatter(q.angle, q.accuracy, s=32, color=COLORS[mol],
                       edgecolor="white", linewidth=0.4, alpha=0.9, zorder=4,
                       label=LABELS[mol])
        else:
            ax.scatter(q.angle, q.accuracy, s=18, color=GREY,
                       edgecolor="white", linewidth=0.3, alpha=0.7, zorder=3)

    ax.axhline(0.5, color="#777777", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("H-bond angle (deg)", fontsize=FONTSIZE_LABEL,
                  color=TEXT_COLOR)
    ax.set_ylabel("mean decoy-pair accuracy across all LLMs",
                  fontsize=FONTSIZE_LABEL, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=FONTSIZE)

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", markersize=5,
                          color=GREY, markeredgecolor="white"))
    labels.append("other mols")
    ax.legend(handles, labels, loc="upper right", fontsize=FONTSIZE - 1,
              frameon=False, handletextpad=0.35, labelspacing=0.45,
              borderaxespad=0.6, markerscale=1.15, handlelength=1.3,
              borderpad=0.2)
    style_ax(ax)
    return save_fig(fig, out, dpi=SAVE_DPI)


def main() -> list[Path]:
    M = geometry_table()
    tsv = OUT / "decoy_hbond_geometry.tsv"
    M.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
    print(f"{len(M)} decoy pairs, {M.molecule.nunique()} molecules")
    return [tsv, plot(M, OUT / "fig3c_decoy_accuracy_vs_angle.png")]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")
