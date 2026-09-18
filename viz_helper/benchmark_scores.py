"""Load geom-cohort benchmark scores from cohort_benchmark_scores.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config.models_config import models_by_id

_REPO = Path(__file__).resolve().parent.parent
COHORT_BENCHMARK_SCORES = _REPO / "cohort_benchmark_scores.json"

BENCHMARK_META_KEYS: frozenset[str] = frozenset({"model", "source"})
SKIP_BENCH_KEYS: frozenset[str] = frozenset({"tau2"})


def parse_pct_score(raw) -> float | None:
    """Parse 0.843, '84.3%', or '~15-20%' → fraction in [0, 1]."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val / 100.0 if val > 1.5 else val
    text = str(raw).strip().replace("~", "").replace("%", "")
    text = text.replace("–", "-").replace("—", "-")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    vals = [float(x) for x in nums]
    mid = sum(vals) / len(vals)
    return mid / 100.0 if mid > 1.5 or "%" in str(raw) else mid


def parse_row_scores(row: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, raw in row.items():
        if key in BENCHMARK_META_KEYS or key in SKIP_BENCH_KEYS:
            continue
        val = parse_pct_score(raw)
        if val is not None:
            scores[key] = val
    return scores


def load_cohort_benchmark_scores(
    scores_path: Path = COHORT_BENCHMARK_SCORES,
    *,
    bench_keys: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """display name → {gpqa, scicode, ...} fractions from cohort_benchmark_scores.json."""
    if not scores_path.is_file():
        return {}
    short_to_display = {m.short: m.display for m in models_by_id().values()}
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    by_display: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for row in payload.get("benchmark_results", []):
        short = row.get("model")
        if not short:
            continue
        display = short_to_display.get(short)
        if display is None:
            skipped.append(f"{short} (unknown short)")
            continue
        parsed = parse_row_scores(row)
        if bench_keys is not None:
            parsed = {k: v for k, v in parsed.items() if k in bench_keys}
        if parsed:
            by_display[display] = parsed
    if by_display:
        print(f"benchmark scores loaded: {len(by_display)} models from {scores_path.name}")
    if skipped:
        print("benchmark scores skipped:", ", ".join(skipped))
    return by_display
