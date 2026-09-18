#!/usr/bin/env python3

"""Math tests for paired LLM−baseline τ differences."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel, wilcoxon

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.paired_vs_baseline import (
    mean_ci_t,
    paired_differences,
    permutation_pvalue_mean,
    summarize_llm_vs_baseline,
    summarize_paired_diff,
)


class TestPairedDifferences(unittest.TestCase):
    def test_elementwise_subtraction(self) -> None:
        d = paired_differences([0.5, 0.2, -0.1], [0.1, 0.2, 0.0])
        np.testing.assert_allclose(d, [0.4, 0.0, -0.1])

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            paired_differences([1.0, 2.0], [1.0])

    def test_nonfinite_raises(self) -> None:
        with self.assertRaises(ValueError):
            paired_differences([1.0, np.nan], [0.0, 0.0])


class TestMeanCiT(unittest.TestCase):
    def test_matches_student_t_formula(self) -> None:
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
        mean, std, se, lo, hi = mean_ci_t(x, alpha=0.05)
        n = len(x)
        self.assertAlmostEqual(mean, float(np.mean(x)))
        self.assertAlmostEqual(std, float(np.std(x, ddof=1)))
        self.assertAlmostEqual(se, std / np.sqrt(n))
        t_crit = float(t_dist.ppf(0.975, df=n - 1))
        self.assertAlmostEqual(lo, mean - t_crit * se)
        self.assertAlmostEqual(hi, mean + t_crit * se)

    def test_n1_ci_collapses_to_mean(self) -> None:
        mean, std, _, lo, hi = mean_ci_t(np.array([0.42]))
        self.assertAlmostEqual(mean, 0.42)
        self.assertEqual(std, 0.0)
        self.assertEqual(lo, hi)


class TestPermutationAndSummary(unittest.TestCase):
    def test_all_zero_diffs_high_p(self) -> None:
        d = np.zeros(20)
        p = permutation_pvalue_mean(d, n_perm=1000, seed=0)
        self.assertGreaterEqual(p, 0.9)
        s = summarize_paired_diff(d, n_perm=500, seed=1)
        self.assertAlmostEqual(s.mean, 0.0)
        self.assertAlmostEqual(s.ci_lo, 0.0)
        self.assertAlmostEqual(s.ci_hi, 0.0)
        self.assertAlmostEqual(s.wilcoxon_pvalue, 1.0)

    def test_strong_positive_shift_small_p(self) -> None:
        rng = np.random.default_rng(0)
        d = 0.4 + 0.05 * rng.standard_normal(25)
        s = summarize_paired_diff(d, n_perm=5000, seed=0)
        self.assertGreater(s.mean, 0.3)
        self.assertGreater(s.ci_lo, 0.0)
        self.assertLess(s.t_pvalue, 1e-6)
        self.assertLess(s.perm_pvalue, 0.01)
        self.assertLess(s.wilcoxon_pvalue, 0.01)

    def test_t_test_matches_scipy(self) -> None:
        d = np.array([0.2, 0.1, 0.3, 0.05, 0.15, -0.02])
        s = summarize_paired_diff(d, run_permutation=False, run_wilcoxon=False)
        ref = ttest_rel(d, np.zeros(len(d)))
        self.assertAlmostEqual(s.t_statistic, float(ref.statistic), places=12)
        self.assertAlmostEqual(s.t_pvalue, float(ref.pvalue), places=12)

    def test_wilcoxon_matches_scipy(self) -> None:
        d = np.array([0.2, 0.1, 0.3, 0.05, 0.15, -0.02])
        s = summarize_paired_diff(d, run_permutation=False, run_wilcoxon=True)
        ref = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        self.assertAlmostEqual(s.wilcoxon_pvalue, float(ref.pvalue), places=12)

    def test_permutation_is_deterministic(self) -> None:
        d = np.array([0.1, -0.05, 0.2, 0.0, 0.15])
        p1 = permutation_pvalue_mean(d, n_perm=2000, seed=42)
        p2 = permutation_pvalue_mean(d, n_perm=2000, seed=42)
        self.assertEqual(p1, p2)

    def test_summarize_llm_vs_baseline_wrapper(self) -> None:
        s = summarize_llm_vs_baseline(
            [0.5, 0.4, 0.6],
            [0.2, 0.2, 0.1],
            n_perm=1000,
            seed=0,
        )
        self.assertEqual(s.n, 3)
        self.assertAlmostEqual(s.mean, np.mean([0.3, 0.2, 0.5]))
        self.assertAlmostEqual(s.median, 0.3)


if __name__ == "__main__":
    unittest.main()
