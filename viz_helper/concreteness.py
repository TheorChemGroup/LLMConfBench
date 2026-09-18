"""Explanation-text concreteness analysis.

Two independent pipelines:

1. REGEX analysis (fast, deterministic): per-explanation counts of H-bond
   vocabulary, quantified measurements and energy mentions, plus vague-term
   counts; writes ``concreteness_vs_tau.tsv`` and the grid/quantile figures.
2. LLM extraction (slow, optional via ``--llm``): asks a local Ollama model to
   split each explanation into SPECIFIC vs VAGUE effects; writes
   ``specific_effects_vs_tau.tsv``.

    python viz_helper/concreteness.py            # regex tables + figures
    python viz_helper/concreteness.py --llm      # also run the LLM extractor
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "viz_helper"))

from config.models_config import models_by_id
from viz.plot_style import save_fig

TEXTS_DIR = Path("texts")
LABELS_TSV = Path("figures/r2scan3c/seed_spread_heatmap_molecules.tsv")
OUT_DIR = Path("figures/r2scan3c")
TABLE_TSV = OUT_DIR / "concreteness_vs_tau.tsv"
RATIO_CURVE_TSV = OUT_DIR / "concreteness_ratio_curve.tsv"
QTY_CURVE_TSV = OUT_DIR / "concreteness_qty_curve.tsv"
SPECIFIC_TSV = OUT_DIR / "specific_effects_vs_tau.tsv"
SEEDS = ("42", "43", "44")


def load_setup(texts_dir: Path = TEXTS_DIR, labels_tsv: Path = LABELS_TSV):
    """(MODELS, MOL_TO_LABEL, MOL_LABELS) with all models that have per-seed texts."""
    available = sorted(p.stem for p in (texts_dir / "seed42").glob("*.jsonl"))
    specs = models_by_id()
    models = {
        mid: (
            specs[mid].display if mid in specs else mid,
            specs[mid].color if mid in specs else "#6b7280",
        )
        for mid in available
    }
    order = [
        line.split("\t")[1]
        for line in labels_tsv.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("label")
    ]
    mol_to_label = {m: i for i, m in enumerate(order, start=1)}
    return models, mol_to_label, set(range(1, len(order) + 1))


ENERGY_RE = re.compile(
    r"\bkcal\b|\bkilocalor\w*|\bkj\b|\bkilojoule\w*|\bhartree\w*|\ba\.u\.|\bE_?h\b",
    re.IGNORECASE,
)
KCAL_RE = re.compile(r"\bkcal\b|\bkilocalor\w*", re.IGNORECASE)
KJ_RE = re.compile(r"\bkj\b|\bkilojoule\w*", re.IGNORECASE)
HARTREE_RE = re.compile(r"\bhartree\w*|\ba\.u\.|\bE_?h\b", re.IGNORECASE)
ENERGY_VAL_RE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*[-–—]\s*(\d+(?:\.\d+)?))?\s*"
    r"(\bkcal\b|\bkilocalor\w*|\bkj\b|\bkilojoule\w*|\bhartree\w*|\ba\.u\.|\bE_?h\b)",
    re.IGNORECASE,
)
QTY_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:Å|å|angstroms?|degrees?|\bdeg\b|°)", re.IGNORECASE)
HBOND_RE = re.compile(r"hydrogen bond|h-?bond|donor|acceptor|h[·.]{1,3}o|o[·.]{1,3}h", re.IGNORECASE)
VAGUE_RE = re.compile(
    r"\bgauche\b|\banti\b|\beclips\w*|\bstagger\w*|\bsteric\w*|"
    r"\bstrain\w*|\brepulsion\w*|\bclash\w*|\bdispersion\w*|"
    r"hyperconjugat\w*|anomeric\w*|\btorsion\w*|\bpucker\w*|"
    r"\bnonbond\w*|\bcrowd\w*",
    re.IGNORECASE,
)
_HARTREE_KCAL = 627.5095
_KJ_KCAL = 0.239006


def _energy_to_kcal(unit: str) -> float:
    u = unit.lower().rstrip(".")
    if u.startswith(("kcal", "kilocalor")):
        return 1.0
    if u.startswith(("kj", "kilojoule")):
        return _KJ_KCAL
    return _HARTREE_KCAL


def energy_values(text: str) -> list[float]:
    """All numeric energy values in the text, converted to kcal/mol (ranges expanded)."""
    out: list[float] = []
    for lo, hi, unit in ENERGY_VAL_RE.findall(text):
        factor = _energy_to_kcal(unit)
        out.append(float(lo) * factor)
        if hi:
            out.append(float(hi) * factor)
    return out


def _reasoning_text(row: dict) -> str:
    rea = row.get("reasoning")
    if isinstance(rea, dict):
        return " ".join(str(v) for v in rea.values() if v)
    return str(rea) if rea else ""


def build_table(
    models: dict,
    mol_to_label: dict,
    mol_labels: set[int],
    *,
    texts_dir: Path = TEXTS_DIR,
    out_tsv: Path = TABLE_TSV,
) -> pd.DataFrame:
    """Count H-bond / quantified / vague terms per explanation and write the TSV."""
    rows = []
    for mid, (name, _color) in models.items():
        for seed in SEEDS:
            for line in (texts_dir / f"seed{seed}" / f"{mid}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["tau"] is None or mol_to_label.get(r["molecule"]) not in mol_labels:
                    continue
                txt = _reasoning_text(r)
                evals = energy_values(txt)
                n_kcal = len(KCAL_RE.findall(txt))
                n_kj = len(KJ_RE.findall(txt))
                n_hartree = len(HARTREE_RE.findall(txt))
                n_energy = n_kcal + n_kj + n_hartree
                n_gt1 = int(sum(v > 1.0 for v in evals))
                n_qty = len(QTY_RE.findall(txt)) + n_energy
                n_hb = len(HBOND_RE.findall(txt))
                n_vague = len(VAGUE_RE.findall(txt))
                concrete = n_hb + n_qty
                rows.append({
                    "model_id": mid, "model": name, "seed": seed,
                    "label": mol_to_label[r["molecule"]], "tau": r["tau"],
                    "n_energy": n_energy, "n_kcal": n_kcal, "n_kj": n_kj,
                    "n_hartree": n_hartree, "n_gt1": n_gt1, "n_qty": n_qty,
                    "n_hbond": n_hb, "n_vague": n_vague, "words": len(txt.split()),
                    "concrete": concrete, "ratio": concrete / (n_vague + 1),
                })
    df = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"{len(df)} explanations from {df.model_id.nunique()} models -> {out_tsv}")
    return df


def _smooth_curve(x, y, frac: float = 0.4, npts: int = 20):
    """Running mean of y over x-sorted points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    n = len(x)
    if n < 6:
        return x, y
    w = max(5, int(frac * n))
    xs, ys = [], []
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        xs.append(float(x[lo:hi].mean()))
        ys.append(float(y[lo:hi].mean()))
    idx = np.unique(np.linspace(0, n - 1, min(npts, n)).astype(int))
    return np.array(xs)[idx], np.array(ys)[idx]


def _tau_order(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    overall = df.groupby("model_id").tau.mean()
    return overall, list(overall.sort_values(ascending=False).index)


def plot_regex_grids(df: pd.DataFrame, models: dict, outdir: Path = OUT_DIR) -> None:
    """Per-model grids (running mean) + the pooled-rho summary figure."""
    overall_tau, order_ids = _tau_order(df)

    def grid(xkey: str, xlab: str, out_png: Path, out_tsv: Path) -> None:
        curve_rows = []
        n = len(order_ids)
        ncol, nrow = 4, math.ceil(n / 4)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, mid in zip(axes, order_ids):
            name, color = models[mid]
            d = df[df.model_id == mid]
            ax.scatter(d[xkey], d.tau, s=12, c=color, alpha=0.45, edgecolors="none")
            xs, ys = _smooth_curve(d[xkey].values, d.tau.values)
            ax.plot(xs, ys, color=color, lw=2.0)
            for xv, yv in zip(xs, ys):
                curve_rows.append({"model_id": mid, "model": name, "x_key": xkey,
                                   "x": round(float(xv), 6), "tau_smooth": round(float(yv), 6)})
            lo, hi = np.quantile(d[xkey], [0.02, 0.98])
            ax.set_xlim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
            ax.set_ylim(-1, 1)
            ax.set_title(f"{name}  (τ̄={overall_tau[mid]:+.2f})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"τ vs {xlab}  (all seeds pooled, models sorted by mean τ)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        save_fig(fig, out_png)
        pd.DataFrame(curve_rows).to_csv(out_tsv, sep="\t", index=False)

    grid("ratio", "concreteness ratio = concrete/(vague+1)",
         outdir / "concreteness_grid_ratio.png", RATIO_CURVE_TSV)
    grid("n_qty", "# quantified measurements (Å / ° / kcal)",
         outdir / "concreteness_grid_qty.png", QTY_CURVE_TSV)

    pooled = []
    for mid in df.model_id.unique():
        d = df[df.model_id == mid]

        def _rho(key: str, d=d) -> float:
            return spearmanr(d[key], d.tau)[0] if d[key].nunique() > 1 else float("nan")

        pooled.append({"model_id": mid, "model": d.model.iloc[0],
                       "rho_ratio": _rho("ratio"), "rho_qty": _rho("n_qty")})
    pooled = pd.DataFrame(pooled).set_index("model_id").loc[order_ids]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, col, lab in ((axes[0], "rho_ratio", "concreteness ratio"),
                         (axes[1], "rho_qty", "quantified measurements")):
        ax.scatter(range(len(order_ids)), pooled[col], s=32, c="#37474f",
                   alpha=0.85, edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="#b0bec5", lw=1)
        ax.set_ylim(-0.3, 0.7)
        ax.set_xticks(range(len(order_ids)))
        ax.set_xticklabels(pooled.model, rotation=90, fontsize=7)
        ax.set_ylabel("overall Spearman ρ(τ, x)")
        ax.set_title(f"ρ(τ, x), seeds pooled — {lab}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, outdir / "concreteness_rho_summary.png")


QS = (0.25, 0.5, 0.75)
FRAC = 0.4
NPTS = 20


def _roll_quantile(x, y, qs=QS, frac=FRAC, npts=NPTS):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    n = len(x)
    if n < 6:
        return x, {q: y for q in qs}
    w = max(5, int(frac * n))
    xs, out = [], {q: [] for q in qs}
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        xs.append(float(x[lo:hi].mean()))
        for q in qs:
            out[q].append(float(np.quantile(y[lo:hi], q)))
    idx = np.unique(np.linspace(0, n - 1, min(npts, n)).astype(int))
    return np.array(xs)[idx], {q: np.array(v)[idx] for q, v in out.items()}


def plot_rolling_quantiles(df: pd.DataFrame, models: dict, outdir: Path = OUT_DIR) -> None:
    overall_tau, order_ids = _tau_order(df)

    def grid(xkey: str, xlab: str, out_png: Path) -> None:
        n = len(order_ids)
        ncol, nrow = 4, math.ceil(n / 4)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, mid in zip(axes, order_ids):
            name, color = models[mid]
            d = df[df.model_id == mid]
            ax.scatter(d[xkey], d.tau, s=10, c=color, alpha=0.25, edgecolors="none")
            xs, q = _roll_quantile(d[xkey].values, d.tau.values)
            ax.fill_between(xs, q[QS[0]], q[QS[2]], color=color, alpha=0.20, lw=0)
            ax.plot(xs, q[QS[1]], color=color, lw=2.0)
            lo, hi = np.quantile(d[xkey], [0.02, 0.98])
            ax.set_xlim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
            ax.set_ylim(-1, 1)
            ax.set_title(f"{name}  (τ̄={overall_tau[mid]:+.2f})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(
            f"τ vs {xlab} — rolling median (q=0.5) ± 25–75% band, seeds pooled",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        save_fig(fig, out_png)

    grid("ratio", "concreteness ratio = concrete/(vague+1)",
         outdir / "concreteness_quantile_ratio.png")
    grid("n_qty", "# quantified measurements (Å / ° / kcal)",
         outdir / "concreteness_quantile_qty.png")


OLLAMA = "http://localhost:11434"
TERM_MODEL = "nuextract:3.8b"
TEMP = 0.2
THINK = False
RETRIES = 8
NUM_PREDICT = 2048
CACHE = Path("texts/specific_effects_cache.json")

PROMPT = """Read the chemistry TEXT below about a molecule's conformers.
Extract the interactions / effects it uses, into two mutually exclusive JSON lists.

"specific": NAMED, well-defined interactions, INCLUDING any quoted number/unit
that appears with them. Examples: "intramolecular H-bond O-H...O", "O...H
distance 1.87 A", "C-H...pi", "1,3-diaxial steric clash", "epoxide ring strain
2.5 kcal/mol", "anti-periplanar lone pair".

"vague": UNSPECIFIED effects invoked WITHOUT a concrete named interaction or
number. Examples: "strain", "steric effects", "electronic effects", "favored",
"destabilized", "gauche" (when no value is tied to it), "dispersion".

RULES: use EXACT substrings from the TEXT; do not invent; deduplicate; put each
phrase in at most one list; return only JSON: {"specific": [...], "vague": [...]}

TEXT:
"""


def _parse_terms(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw).replace("```", "")
    obj = None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            obj = None
    if not isinstance(obj, dict):
        def arr(key):
            mm = re.search(r'"' + key + r'"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
            return re.findall(r'"((?:[^"\\]|\\.)*)"', mm.group(1)) if mm else []
        obj = {"specific": arr("specific"), "vague": arr("vague")}
    return ([str(x) for x in obj.get("specific", [])],
            [str(x) for x in obj.get("vague", [])])


def extract_terms(text: str, *, model: str = TERM_MODEL):
    import requests

    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": PROMPT + text, "stream": False,
              "think": THINK,
              "options": {"temperature": TEMP, "num_predict": NUM_PREDICT}},
        timeout=900,
    )
    r.raise_for_status()
    data = r.json()
    sp, vg = _parse_terms(data.get("response"))
    if not (sp or vg):
        sp, vg = _parse_terms(data.get("thinking"))
    return sp, vg


def run_llm_extraction(
    models: dict,
    mol_to_label: dict,
    mol_labels: set[int],
    *,
    texts_dir: Path = TEXTS_DIR,
    which: list[str] | None = None,
    max_explanations: int | None = None,
    cache_path: Path = CACHE,
    force: bool = False,
    model: str = TERM_MODEL,
) -> pd.DataFrame:
    """LLM-extract specific vs vague effects per explanation (cached)."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if (
        cache_path.is_file() and not force) else {}

    def terms_for(mid: str, mol: str, seed: str, text: str):
        key = f"{mid}|{mol}|{seed}"
        got = cache.get(key)
        if got and (got.get("specific") or got.get("vague")):
            return got["specific"], got["vague"]
        sp, vg = [], []
        for _ in range(RETRIES):
            sp, vg = extract_terms(text, model=model)
            if sp or vg:
                cache[key] = {"specific": sp, "vague": vg}
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                return sp, vg
        return sp, vg

    rows = []
    for mid in (which or list(models)):
        name = models[mid][0]
        for seed in SEEDS:
            for line in (texts_dir / f"seed{seed}" / f"{mid}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["tau"] is None or mol_to_label.get(r["molecule"]) not in mol_labels:
                    continue
                txt = _reasoning_text(r)
                sp, vg = terms_for(mid, r["molecule"], seed, txt)
                rows.append({"model_id": mid, "model": name, "seed": seed,
                             "label": mol_to_label[r["molecule"]], "tau": r["tau"],
                             "n_specific": len(sp), "n_vague": len(vg),
                             "ratio_spec": len(sp) / (len(vg) + 1),
                             "words": len(txt.split())})
                if max_explanations is not None and len(rows) >= max_explanations:
                    break
            if max_explanations is not None and len(rows) >= max_explanations:
                break
        if max_explanations is not None and len(rows) >= max_explanations:
            break

    df = pd.DataFrame(rows)
    df.to_csv(SPECIFIC_TSV, sep="\t", index=False)
    print(f"{len(df)} explanations extracted; cache -> {cache_path}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="also run the LLM extractor")
    ap.add_argument("--llm-model", default=TERM_MODEL)
    ap.add_argument("--force", action="store_true", help="ignore the LLM cache")
    args = ap.parse_args()

    models, mol_to_label, mol_labels = load_setup()
    print(f"{len(models)} models x {len(mol_labels)} molecules")

    df = build_table(models, mol_to_label, mol_labels)
    plot_regex_grids(df, models)
    plot_rolling_quantiles(df, models)

    if args.llm:
        run_llm_extraction(models, mol_to_label, mol_labels,
                           force=args.force, model=args.llm_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
