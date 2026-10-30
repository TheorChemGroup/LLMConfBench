#!/usr/bin/env python3

"""Mathematical correctness tests for paper metrics (τ@floor, CI, min@k)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from scipy.stats import t as t_dist

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_CCSDT = _REPO / "orca_ccsdt"
if str(_CCSDT) not in sys.path:
    sys.path.insert(0, str(_CCSDT))

from compare_baselines_subset import (  # type: ignore
    reliable_tau as conf_reliable_tau,
)

from metrics.conformer_min_in_topk import letters_at_global_minimum, min_in_topk
from metrics.grouped_metrics import mean_std_ci
from metrics.kendall_tau_ranking import augment_meta
from metrics.option2_metrics import (
    label_sort_key,
    reliable_pairs,
    reliable_tau,
    score_molecule,
)


class TestReliableTauMath(unittest.TestCase):
    def test_floor_zero_matches_scipy_kendall_no_ties(self) -> None:
        de = np.array([0.0, 1.5, 3.0, 4.5, 7.0])
        pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        tau, n = reliable_tau(de, pred, floor=0.0)
        true_rank = np.argsort(np.argsort(de))
        scipy_tau = float(kendalltau(true_rank, pred).statistic)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(tau, scipy_tau, places=12)
        self.assertAlmostEqual(tau, 1.0)

    def test_one_discordant_pair(self) -> None:
        de = np.array([0.0, 1.0, 2.0, 3.0])
        pred = np.array([0.0, 2.0, 1.0, 3.0])
        tau, n = reliable_tau(de, pred, floor=0.0)
        self.assertEqual(n, 6)
        self.assertAlmostEqual(tau, 4.0 / 6.0)

    def test_floor_boundary_uses_ge(self) -> None:
        de = np.array([0.0, 1.0, 3.0])
        pred = np.array([0.0, 1.0, 2.0])
        i, _j = reliable_pairs(de, floor=1.0)
        self.assertEqual(len(i), 3)
        tau, n = reliable_tau(de, pred, floor=1.0)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(tau, 1.0)
        _, n_gt = reliable_tau(de, pred, floor=1.0 + 1e-12)
        self.assertEqual(n_gt, 2)

    def test_paper_floor_2(self) -> None:
        """Manuscript primary floor |ΔE| ≥ 2.0."""
        de = np.array([0.0, 1.0, 2.0, 4.0])
        pred = np.array([0.0, 1.0, 2.0, 3.0])
        tau, n = reliable_tau(de, pred, floor=2.0)

        self.assertEqual(n, 4)
        self.assertAlmostEqual(tau, 1.0)

    def test_option2_matches_conf_index_form(self) -> None:
        """option2 array API ≡ compare_baselines_subset dict API."""
        letter_de = {"1": 0.0, "2": 0.8, "3": 2.5, "4": 4.0}
        ranking = ["1", "3", "2", "4"]
        letters = sorted(letter_de.keys(), key=label_sort_key)
        de = np.array([letter_de[L] for L in letters], dtype=float)
        pred_rank = np.array([ranking.index(L) for L in letters], dtype=float)
        tau_arr, n_arr = reliable_tau(de, pred_rank, floor=1.0)

        energy = {int(L) - 1: letter_de[L] for L in letters}
        conf_ranking = [int(L) - 1 for L in ranking]
        tau_conf = conf_reliable_tau(energy, conf_ranking, 1.0)
        self.assertIsNotNone(tau_conf)
        self.assertEqual(n_arr, 5)
        self.assertAlmostEqual(tau_arr, float(tau_conf), places=12)


class TestScoreMoleculeTau(unittest.TestCase):
    def test_reversed_has_negative_floor_tau(self) -> None:
        letter_de = {str(i): float(i) for i in range(1, 6)}
        ranking = ["5", "4", "3", "2", "1"]
        row = score_molecule(ranking, letter_de, floors=[2.0])
        self.assertAlmostEqual(row["tau_full"], -1.0)
        self.assertAlmostEqual(row["tau_f2.0"], -1.0)


class TestMinInTopK(unittest.TestCase):
    """Seed-spread figure uses min@10 hit coloring."""

    def test_letters_at_global_minimum(self) -> None:
        self.assertEqual(letters_at_global_minimum({"a": 1.0, "b": 0.0, "c": 0.0}), {"b", "c"})

    def test_hit_and_miss(self) -> None:
        ranking = ["a", "b", "c", "d"]
        self.assertTrue(min_in_topk(ranking, {"c"}, 3))
        self.assertFalse(min_in_topk(ranking, {"c"}, 2))
        self.assertTrue(min_in_topk(ranking, {"a", "z"}, 1))

    def test_k_larger_than_ranking(self) -> None:
        self.assertTrue(min_in_topk(["a", "b"], {"b"}, 99))


class TestMeanStdCi(unittest.TestCase):
    def test_known_t_interval(self) -> None:
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std, se, lo, hi = mean_std_ci(vals)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, float(np.std(vals, ddof=1)))
        self.assertAlmostEqual(se, std / math.sqrt(5))
        tcrit = float(t_dist.ppf(0.975, df=4))
        self.assertAlmostEqual(lo, mean - tcrit * se)
        self.assertAlmostEqual(hi, mean + tcrit * se)
        self.assertAlmostEqual(0.5 * (hi - lo), tcrit * se)

    def test_single_value(self) -> None:
        mean, std, se, lo, hi = mean_std_ci(np.array([2.5]))
        self.assertEqual((mean, std, se, lo, hi), (2.5, 0.0, 0.0, 2.5, 2.5))


class TestAugmentMeta(unittest.TestCase):
    """r2SCAN-3c paper scores replace xTB ΔE via conf-index override."""

    def test_override_maps_by_conf_index(self) -> None:
        meta = {
            "mol": {
                "molecule": "mol",
                "public_label_to_conf_index": {"1": 0, "2": 1, "3": 2},
                "letter_to_deltaE_kcal_mol": {"1": 9.0, "2": 8.0, "3": 7.0},
            }
        }
        override = {"mol": {0: 0.0, 1: 1.5, 2: 3.0}}
        out = augment_meta(meta, override, strict=True)
        self.assertEqual(out["mol"]["letter_to_deltaE_kcal_mol"], {"1": 0.0, "2": 1.5, "3": 3.0})

    def test_strict_drops_incomplete(self) -> None:
        meta = {
            "mol": {
                "molecule": "mol",
                "public_label_to_conf_index": {"1": 0, "2": 1},
                "letter_to_deltaE_kcal_mol": {"1": 0.0, "2": 1.0},
            }
        }
        override = {"mol": {0: 0.0}}
        out = augment_meta(meta, override, strict=True)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
