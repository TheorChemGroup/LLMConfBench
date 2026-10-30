#!/usr/bin/env python3

"""Paired molecule-level comparisons of LLM τ against a force-field baseline.

For each molecule i:
    d_i = τ_i(LLM) − τ_i(baseline)

Summary: mean / median of d, Student t 95% CI for the mean, two-sided paired
t-test, optional sign-flip permutation p-value, and Wilcoxon signed-rank p.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel, wilcoxon


@dataclass(frozen=True)
class PairedDiffSummary:
    n: int
    mean: float
    median: float
    std: float
    se: float
    ci_lo: float
    ci_hi: float
    t_statistic: float
    t_pvalue: float
    perm_pvalue: float | None
    wilcoxon_pvalue: float | None


def paired_differences(
    llm_taus: np.ndarray | list[float],
    baseline_taus: np.ndarray | list[float],
) -> np.ndarray:
    """Per-molecule d_i = τ_LLM − τ_baseline (same length, finite values)."""
    a = np.asarray(llm_taus, dtype=float)
    b = np.asarray(baseline_taus, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: llm {a.shape} vs baseline {b.shape}")
    if a.ndim != 1:
        raise ValueError("taus must be 1-D")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("taus must be finite")
    return a - b


def mean_ci_t(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float, float]:
    """Mean, sample SD, SE, and two-sided t CI for the mean."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0, 0.0, mean, mean
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df=n - 1))
    return mean, std, se, mean - t_crit * se, mean + t_crit * se


def permutation_pvalue_mean(
    diffs: np.ndarray,
    *,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation test of H0: mean(d) = 0."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(diffs)))
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rng = np.random.default_rng(seed)

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null_means = np.abs(np.mean(signs * diffs, axis=1))

    return float((1 + np.sum(null_means >= observed)) / (n_perm + 1))


def summarize_paired_diff(
    diffs: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
    n_perm: int = 10_000,
    seed: int = 0,
    run_permutation: bool = True,
    run_wilcoxon: bool = True,
) -> PairedDiffSummary:
    """Aggregate paired differences d_i into mean CI and hypothesis tests."""
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1:
        raise ValueError("diffs must be 1-D")
    if len(d) == 0:
        raise ValueError("diffs is empty")
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs must be finite")

    mean, std, se, ci_lo, ci_hi = mean_ci_t(d, alpha=alpha)
    median = float(np.median(d))
    n = len(d)

    if n < 2:
        t_stat, t_p = float("nan"), float("nan")
    else:
        t_res = ttest_rel(d, np.zeros(n))
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    perm_p: float | None = None
    if run_permutation and n >= 1:
        perm_p = permutation_pvalue_mean(d, n_perm=n_perm, seed=seed)

    w_p: float | None = None
    if run_wilcoxon and n >= 1:

        if np.allclose(d, 0.0):
            w_p = 1.0
        else:
            w_res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            w_p = float(w_res.pvalue)

    return PairedDiffSummary(
        n=n,
        mean=mean,
        median=median,
        std=std,
        se=se,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        t_statistic=t_stat,
        t_pvalue=t_p,
        perm_pvalue=perm_p,
        wilcoxon_pvalue=w_p,
    )


def summarize_llm_vs_baseline(
    llm_taus: np.ndarray | list[float],
    baseline_taus: np.ndarray | list[float],
    **kwargs,
) -> PairedDiffSummary:
    return summarize_paired_diff(paired_differences(llm_taus, baseline_taus), **kwargs)
