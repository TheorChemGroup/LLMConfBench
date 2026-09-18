#!/usr/bin/env python3

"""Math tests for τ versus release-date trend summaries."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.tau_vs_release_trend import (
    leave_one_family_out,
    parse_release_date,
    summarize_tau_vs_date,
)


class TestParseDate(unittest.TestCase):
    def test_iso_string(self) -> None:
        self.assertEqual(parse_release_date("2025-08-07"), date(2025, 8, 7))


class TestSummarizeTrend(unittest.TestCase):
    def test_perfect_linear_r2(self) -> None:
        dates = [date(2024, 1, 1), date(2025, 1, 1), date(2026, 1, 1), date(2027, 1, 1)]

        x = np.array([d.toordinal() for d in dates], dtype=float)
        y = 0.1 * (x - x[0]) / 365.25
        s = summarize_tau_vs_date(y, dates)
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.r_squared, 1.0, places=12)
        self.assertAlmostEqual(s.pearson_r, 1.0, places=12)
        self.assertAlmostEqual(s.spearman_rho, 1.0, places=12)

    def test_matches_scipy(self) -> None:
        dates = [
            date(2024, 9, 19),
            date(2025, 1, 20),
            date(2025, 8, 7),
            date(2026, 2, 23),
            date(2026, 7, 10),
        ]
        y = np.array([0.0, 0.01, 0.32, 0.19, 0.62])
        x = np.array([d.toordinal() for d in dates], dtype=float)
        s = summarize_tau_vs_date(y, dates)
        rho, sp = spearmanr(x, y)
        r, pp = pearsonr(x, y)
        fit = linregress(x, y)
        self.assertAlmostEqual(s.spearman_rho, float(rho), places=12)
        self.assertAlmostEqual(s.spearman_p, float(sp), places=12)
        self.assertAlmostEqual(s.pearson_r, float(r), places=12)
        self.assertAlmostEqual(s.pearson_p, float(pp), places=12)
        self.assertAlmostEqual(s.r_squared, float(fit.rvalue) ** 2, places=12)

    def test_too_few_points_raises(self) -> None:
        with self.assertRaises(ValueError):
            summarize_tau_vs_date([0.1, 0.2], ["2025-01-01", "2026-01-01"])


class TestLeaveOneFamilyOut(unittest.TestCase):
    def test_dropping_family_changes_n(self) -> None:
        dates = [
            "2024-01-01",
            "2024-06-01",
            "2025-01-01",
            "2025-06-01",
            "2026-01-01",
            "2026-06-01",
        ]
        taus = [0.0, 0.05, 0.1, 0.4, 0.5, 0.55]
        fams = ["A", "A", "B", "B", "C", "C"]
        out = leave_one_family_out(taus, dates, fams)
        self.assertEqual(out["A"].n, 4)
        self.assertEqual(out["B"].n, 4)
        self.assertEqual(out["C"].n, 4)


if __name__ == "__main__":
    unittest.main()
