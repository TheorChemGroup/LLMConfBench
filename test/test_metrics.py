#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.kendall_tau_ranking import kendall_for_molecule
from metrics.option2_metrics import reliable_tau, score_molecule
from metrics.random_baseline import min_at_k_random


class TestReliableTau(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        de = np.array([0.0, 1.0, 3.0, 5.0])
        pred = np.array([0.0, 1.0, 2.0, 3.0])
        tau, n = reliable_tau(de, pred, floor=1.0)
        self.assertEqual(n, 6)
        self.assertAlmostEqual(tau, 1.0)

    def test_reversed_ranking(self) -> None:
        de = np.array([0.0, 2.0, 4.0])
        pred = np.array([2.0, 1.0, 0.0])
        tau, n = reliable_tau(de, pred, floor=1.0)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(tau, -1.0)

    def test_floor_excludes_small_gaps(self) -> None:
        de = np.array([0.0, 0.4, 5.0])
        pred = np.array([2.0, 1.0, 0.0])
        tau_lo, n_lo = reliable_tau(de, pred, floor=0.5)
        tau_hi, n_hi = reliable_tau(de, pred, floor=1.0)
        self.assertEqual(n_lo, 2)
        self.assertEqual(n_hi, 2)
        self.assertAlmostEqual(tau_lo, -1.0)
        self.assertAlmostEqual(tau_hi, -1.0)

    def test_no_pairs_returns_nan(self) -> None:
        de = np.array([0.0, 0.1, 0.2])
        pred = np.array([0.0, 1.0, 2.0])
        tau, n = reliable_tau(de, pred, floor=1.0)
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(tau))


class TestScoreMolecule(unittest.TestCase):
    def _letter_de(self) -> dict[str, float]:
        return {str(i): float(i - 1) for i in range(1, 6)}

    def test_valid_perfect_ranking(self) -> None:
        letters = self._letter_de()
        ranking = ["1", "2", "3", "4", "5"]
        row = score_molecule(ranking, letters, floors=[1.0])
        self.assertTrue(row["valid"])
        self.assertAlmostEqual(row["tau_full"], 1.0)
        self.assertAlmostEqual(row["top1"], 1.0)
        self.assertAlmostEqual(row["tau_f1.0"], 1.0)

    def test_invalid_permutation(self) -> None:
        row = score_molecule(["1", "2", "3"], self._letter_de(), floors=[1.0])
        self.assertFalse(row["valid"])


class TestKendallForMolecule(unittest.TestCase):
    def test_matches_option2_full_tau(self) -> None:
        letter_de = {str(i): float((i - 1) ** 2) for i in range(1, 8)}
        ranking = sorted(letter_de.keys(), key=lambda L: letter_de[L])
        tau_k, _ = kendall_for_molecule(letter_de, ranking)
        row = score_molecule(ranking, letter_de, floors=[1.0])
        self.assertAlmostEqual(tau_k, row["tau_full"], places=6)

    def test_partial_ranking_raises(self) -> None:
        letter_de = {"1": 0.0, "2": 1.0, "3": 2.0}
        with self.assertRaises(ValueError):
            kendall_for_molecule(letter_de, ["1", "2"])


class TestRandomBaseline(unittest.TestCase):
    def test_min_at_k_single_minimum(self) -> None:
        self.assertAlmostEqual(min_at_k_random(10, 1, 3), 0.3)

    def test_min_at_k_all_in_topk(self) -> None:
        self.assertAlmostEqual(min_at_k_random(5, 2, 10), 1.0)


class TestConfig(unittest.TestCase):
    def test_geom_benchmark_models_are_complete(self) -> None:
        from config.models_config import models_in_group

        models = models_in_group("geom_benchmark")
        self.assertGreater(len(models), 0)
        self.assertEqual(len({m.id for m in models}), len(models))
        self.assertEqual(len({m.display for m in models}), len(models))
        self.assertTrue(all(m.answers for m in models))
        self.assertTrue(all(m.color.startswith("#") for m in models))

    def test_baseline_resolution(self) -> None:
        from config.models_config import resolve_deltae_json

        self.assertIsNone(resolve_deltae_json("gfn2-xtb"))
        p = resolve_deltae_json("r2scan3c")
        self.assertIsNotNone(p)
        self.assertTrue(p.name.endswith(".json"))

    def test_release_dates_from_models_yaml(self) -> None:
        from config.models_config import release_dates_by_display

        dates = release_dates_by_display()
        self.assertIn("Claude Opus 5", dates)
        self.assertEqual(dates["Claude Opus 5"].strftime("%Y-%m-%d"), "2026-07-24")


if __name__ == "__main__":
    unittest.main()
