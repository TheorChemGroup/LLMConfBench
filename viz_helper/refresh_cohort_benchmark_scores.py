"""Refresh cohort_benchmark_scores.json from OpenRouter + Modelgrep (max per field).

For each geom-cohort model and benchmark field, keeps the higher score from:
  1. OpenRouter rankings API (GPQA Diamond)
  2. Modelgrep API (Artificial Analysis fields)

  python viz_helper/refresh_cohort_benchmark_scores.py
  python viz_helper/refresh_cohort_benchmark_scores.py --openrouter-json openrouter_export.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scores import COHORT_BENCHMARK_SCORES

from config.cohort_model_slugs import COHORT_MODEL_SLUGS
from config.models_config import models_by_id

OPENROUTER_EXPORT = _REPO / "openrouter_export.json"
OPENROUTER_BENCHMARKS_API = "https://openrouter.ai/api/v1/benchmarks"
MODELGREP_API = "https://modelgrep.com/api/v1/models"

OPENROUTER_BENCHMARK_TYPE_TO_KEY: dict[str, str] = {
    "gpqa_diamond": "gpqa",
}

MG_SKIP_KEYS: frozenset[str] = frozenset({
    "tau2",
    "tps",
    "ttft",
    "agentic_pct",
    "coding_pct",
    "intelligence_pct",
    "math_pct",
    "terminalbench_pct",
})

OPENROUTER_SLUG_CANDIDATES: dict[str, list[str]] = {
    "claude-opus-5": ["anthropic/claude-opus-5"],
    "claude-sonnet-5": ["anthropic/claude-sonnet-5"],
    "claude-sonnet-4.5": ["anthropic/claude-4.5-sonnet", "anthropic/claude-sonnet-4.5"],
    "claude-sonnet-4": ["anthropic/claude-4-sonnet", "anthropic/claude-sonnet-4"],
    "gpt-5.6-sol": ["openai/gpt-5.6-sol"],
    "gpt-6-astra": ["openai/gpt-6-astra"],
    "gpt-5": ["openai/gpt-5"],
    "o4-mini": ["openai/o4-mini"],
    "gemini-3.6-flash": ["google/gemini-3.6-flash"],
    "gemini-3-flash-preview": ["google/gemini-3-flash-preview"],
    "gemini-2.5-flash": ["google/gemini-2.5-flash"],
    "gemma-3-27b": ["google/gemma-3-27b-it"],
    "gemma-4-31b": ["google/gemma-4-31b-it"],
    "kimi-k3": ["moonshotai/kimi-k3"],
    "kimi-k2-thinking": ["moonshotai/kimi-k2-thinking"],
    "qwen3.8-27b": ["qwen/qwen3.8-27b"],
    "qwen3.7-max": ["qwen/qwen3.7-max"],
    "qwen3.6-27b": ["qwen/qwen3.6-27b"],
    "qwen3.5-27b": ["qwen/qwen3.5-27b"],
    "qwen2.5-32b": ["qwen/qwen2.5-32b", "qwen/qwen2.5-32b-instruct"],
    "qwen3-32b": ["qwen/qwen3-32b", "qwen/qwen3-32b-04-28"],
    "qwen3.5-35b-a3b": ["qwen/qwen3.5-35b-a3b"],
    "qwen3.6-35b-a3b": ["qwen/qwen3.6-35b-a3b"],
    "deepseek-v4-flash-0731": [
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-flash",
    ],
    "deepseek-v3.2": ["deepseek/deepseek-v3.2"],
    "deepseek-r1-32b": [
        "deepseek/deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-r1",
    ],
    "muse-glimmer-30b": ["meta/muse-glimmer-30b"],
}


def slug_base(slug: str) -> str:
    s = (slug or "").lower()
    s = re.sub(r"-\d{8}$", "", s)
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    return s


def slug_candidates(short: str, registry_slug: str | None) -> list[str]:
    cands = list(OPENROUTER_SLUG_CANDIDATES.get(short, []))
    if registry_slug:
        cands = [registry_slug] + [
            c for c in cands if slug_base(c) != slug_base(registry_slug)
        ]
    seen: set[str] = set()
    out: list[str] = []
    for cand in cands:
        base = slug_base(cand)
        if base and base not in seen:
            seen.add(base)
            out.append(cand)
    return out


def normalize_score(raw) -> float | None:
    if raw is None:
        return None
    val = float(raw)
    return val / 100.0 if val > 1.5 else val


def fmt_score(val: float) -> str | float:
    v = float(val)
    if v > 1.5:
        return f"{v:.1f}%"
    return round(v, 4)


def fetch_openrouter_benchmarks(
    *,
    benchmark_type: str | None = None,
    api_key: str | None = None,
) -> dict:
    params = ["source=openrouter"]
    if benchmark_type:
        params.append(f"benchmark_type={benchmark_type}")
    url = f"{OPENROUTER_BENCHMARKS_API}?{'&'.join(params)}"
    token = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not token:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment")
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "refresh_cohort_benchmark_scores/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def fetch_modelgrep(limit: int = 200) -> dict[str, dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        url = f"{MODELGREP_API}?limit={limit}&offset={offset}"
        req = urllib.request.Request(url, headers={"User-Agent": "refresh_cohort_benchmark_scores/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        rows.extend(payload.get("data") or [])
        meta = payload.get("meta") or {}
        if not meta.get("has_more"):
            break
        offset = int(meta["next_offset"])
    by_id: dict[str, dict] = {}
    for row in rows:
        mid = row.get("id") or ""
        if ":batch" in mid:
            continue
        by_id[mid] = row
    return by_id


def load_openrouter_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index_openrouter_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    by_type: dict[str, dict[str, dict]] = {}
    for row in rows:
        bench_type = row.get("benchmark_type")
        slug = row.get("model_permaslug") or ""
        base = slug_base(slug)
        if not bench_type or not base:
            continue
        by_type.setdefault(str(bench_type), {})[base] = row
    return by_type


def find_openrouter_row(
    short: str,
    registry_slug: str | None,
    by_base: dict[str, dict],
) -> dict | None:
    for cand in slug_candidates(short, registry_slug):
        row = by_base.get(slug_base(cand))
        if row is not None:
            return row
    return None


def openrouter_row_score(row: dict | None) -> float | None:
    if row is None:
        return None
    raw = row.get("accuracy")
    if raw is None:
        raw = row.get("primary_score")
    return normalize_score(raw)


def openrouter_scores_by_display(
    payload: dict,
) -> dict[str, dict[str, float]]:
    rows = payload.get("data") or []
    by_type = index_openrouter_rows(rows)
    display_to_short = {m.display: m.short for m in models_by_id().values()}
    out: dict[str, dict[str, float]] = {}
    for display, registry_slug in COHORT_MODEL_SLUGS.items():
        short = display_to_short.get(display)
        if not short:
            continue
        scores: dict[str, float] = {}
        for bench_type, key in OPENROUTER_BENCHMARK_TYPE_TO_KEY.items():
            row = find_openrouter_row(short, registry_slug, by_type.get(bench_type, {}))
            val = openrouter_row_score(row)
            if val is not None:
                scores[key] = val
        if scores:
            out[display] = scores
    return out


def modelgrep_scores_by_display(mg_by_id: dict[str, dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for display, registry_slug in COHORT_MODEL_SLUGS.items():
        if not registry_slug:
            continue
        row = mg_by_id.get(registry_slug)
        if not row:
            continue
        aa = (row.get("benchmarks") or {}).get("artificial_analysis") or {}
        scores: dict[str, float] = {}
        for key, raw in aa.items():
            if key in MG_SKIP_KEYS or raw is None:
                continue
            val = normalize_score(raw)
            if val is not None:
                scores[key] = val
        if scores:
            out[display] = scores
    return out


def merge_max_scores(
    openrouter: dict[str, dict[str, float]],
    modelgrep: dict[str, dict[str, float]],
) -> tuple[list[dict], list[str]]:
    display_to_short = {m.display: m.short for m in models_by_id().values()}
    results: list[dict] = []
    dropped: list[str] = []

    for display in sorted(COHORT_MODEL_SLUGS):
        short = display_to_short.get(display)
        if not short:
            continue
        or_scores = openrouter.get(display, {})
        mg_scores = modelgrep.get(display, {})
        keys = sorted(set(or_scores) | set(mg_scores))
        if not keys:
            dropped.append(display)
            continue

        merged: dict[str, float] = {}
        sources: dict[str, str] = {}
        for key in keys:
            or_val = or_scores.get(key)
            mg_val = mg_scores.get(key)
            if or_val is None and mg_val is None:
                continue
            if or_val is None:
                merged[key] = mg_val
                sources[key] = "modelgrep"
            elif mg_val is None or or_val >= mg_val:
                merged[key] = or_val
                sources[key] = "openrouter"
            else:
                merged[key] = mg_val
                sources[key] = "modelgrep"

        if not merged:
            dropped.append(display)
            continue

        row: dict[str, str | float] = {
            "model": short,
            "source": "; ".join(
                f"{src}: {', '.join(k for k, s in sorted(sources.items()) if s == src)}"
                for src in ("openrouter", "modelgrep")
                if any(s == src for s in sources.values())
            ),
        }
        for key, val in sorted(merged.items()):
            row[key] = fmt_score(val)
        results.append(row)

    return results, dropped


def refresh_cohort_benchmark_scores(
    *,
    openrouter_payload: dict | None = None,
    openrouter_path: Path | None = None,
    mg_by_id: dict[str, dict] | None = None,
) -> tuple[list[dict], list[str]]:
    if openrouter_payload is None:
        if openrouter_path and openrouter_path.is_file():
            openrouter_payload = load_openrouter_payload(openrouter_path)
        else:
            openrouter_payload = fetch_openrouter_benchmarks()
    mg = mg_by_id if mg_by_id is not None else fetch_modelgrep()
    or_scores = openrouter_scores_by_display(openrouter_payload)
    mg_scores = modelgrep_scores_by_display(mg)
    return merge_max_scores(or_scores, mg_scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--openrouter-json",
        type=Path,
        default=None,
        help=f"Local OpenRouter API dump (default: fetch live or use {OPENROUTER_EXPORT.name})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=COHORT_BENCHMARK_SCORES,
        help=f"Output JSON (default: {COHORT_BENCHMARK_SCORES.name})",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    or_path = args.openrouter_json or (OPENROUTER_EXPORT if OPENROUTER_EXPORT.is_file() else None)
    openrouter_payload = load_openrouter_payload(or_path) if or_path else None
    results, dropped = refresh_cohort_benchmark_scores(
        openrouter_payload=openrouter_payload,
        openrouter_path=None if openrouter_payload is not None else or_path,
    )
    payload = {
        "note": (
            "Geom-cohort benchmark scores. Per field: max(OpenRouter, Modelgrep). "
            "Regenerate with viz_helper/refresh_cohort_benchmark_scores.py."
        ),
        "benchmark_results": results,
    }
    if not args.dry_run:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if openrouter_payload is not None and or_path != args.out:
            OPENROUTER_EXPORT.write_text(
                json.dumps(openrouter_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    print(f"{'would write' if args.dry_run else 'wrote'} {len(results)} models to {args.out}")
    if dropped:
        print(f"  dropped (no scores from either source): {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
