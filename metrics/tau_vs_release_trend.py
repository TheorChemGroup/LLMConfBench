#!/usr/bin/env python3

"""Association of model mean τ with release date (Spearman, Pearson, R²)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr


@dataclass(frozen=True)
class TrendSummary:
    n: int
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    r_squared: float
    slope_per_year: float
    intercept: float


def parse_release_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if " " in text:
        text = text.split(" ", 1)[0]
    return date.fromisoformat(text)


def dates_to_ordinal(dates: list[date] | np.ndarray) -> np.ndarray:
    return np.asarray([d.toordinal() for d in dates], dtype=float)


def summarize_tau_vs_date(
    taus: np.ndarray | list[float],
    dates: list[date] | list[str] | np.ndarray,
) -> TrendSummary:
    """Spearman / Pearson / OLS R² for mean τ versus release date."""
    y = np.asarray(taus, dtype=float)
    if y.ndim != 1:
        raise ValueError("taus must be 1-D")
    parsed = [parse_release_date(d) for d in dates]
    if len(parsed) != len(y):
        raise ValueError(f"length mismatch: {len(y)} taus vs {len(parsed)} dates")
    if len(y) < 3:
        raise ValueError("need at least 3 points")
    if not np.all(np.isfinite(y)):
        raise ValueError("taus must be finite")

    x = dates_to_ordinal(parsed)
    rho, sp = spearmanr(x, y)
    r, pp = pearsonr(x, y)
    fit = linregress(x, y)

    slope_per_year = float(fit.slope) * 365.25
    return TrendSummary(
        n=len(y),
        spearman_rho=float(rho),
        spearman_p=float(sp),
        pearson_r=float(r),
        pearson_p=float(pp),
        r_squared=float(fit.rvalue) ** 2,
        slope_per_year=slope_per_year,
        intercept=float(fit.intercept),
    )


def leave_one_family_out(
    taus: np.ndarray | list[float],
    dates: list[date] | list[str],
    families: list[str],
) -> dict[str, TrendSummary]:
    """Refit after dropping each family in turn (key = dropped family)."""
    y = np.asarray(taus, dtype=float)
    fam = list(families)
    if len(fam) != len(y):
        raise ValueError("families length must match taus")
    out: dict[str, TrendSummary] = {}
    for dropped in sorted(set(fam)):
        keep = [i for i, f in enumerate(fam) if f != dropped]
        if len(keep) < 3:
            continue
        out[dropped] = summarize_tau_vs_date(
            y[keep],
            [dates[i] for i in keep],
        )
    return out
