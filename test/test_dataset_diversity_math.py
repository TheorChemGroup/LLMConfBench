#!/usr/bin/env python3

"""Math correctness for viz_helper/plot_dataset_diversity_panels.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.spatial.distance import squareform

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_VIZ = _REPO / "viz_helper"
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

from plot_dataset_diversity_panels import (
    KB_T_KCAL,
    condensed_pair_count,
    ensemble_heavy_rmsd_stats,
    ensemble_tfd_stats,
    heavy_atom_indices,
    kb_t_kcal,
    pairwise_mean_max,
    tanimoto_distance_matrix,
)

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _embed_confs(smiles: str, n_confs: int = 4, seed: int = 1) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=50)
    return mol


class TestKbT(unittest.TestCase):
    def test_room_temperature(self) -> None:
        self.assertAlmostEqual(kb_t_kcal(298.15), 0.5924849497137634, places=10)
        self.assertAlmostEqual(KB_T_KCAL, kb_t_kcal(298.15), places=12)

    def test_scales_linearly_with_t(self) -> None:
        self.assertAlmostEqual(kb_t_kcal(596.3), 2 * kb_t_kcal(298.15), places=10)


class TestTanimotoDistance(unittest.TestCase):
    def test_identical_fingerprints_distance_zero(self) -> None:
        fp = MORGAN.GetFingerprint(Chem.MolFromSmiles("CCO"))
        dist = tanimoto_distance_matrix([fp, fp, fp])
        np.testing.assert_allclose(dist, np.zeros((3, 3)))

    def test_distance_is_one_minus_tc(self) -> None:
        fps = [
            MORGAN.GetFingerprint(Chem.MolFromSmiles(s))
            for s in ("CCO", "CCN", "c1ccccc1")
        ]
        dist = tanimoto_distance_matrix(fps)
        for i in range(3):
            self.assertEqual(dist[i, i], 0.0)
            for j in range(i + 1, 3):
                tc = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                self.assertAlmostEqual(dist[i, j], 1.0 - tc, places=12)
                self.assertEqual(dist[i, j], dist[j, i])

    def test_values_in_unit_interval(self) -> None:
        fps = [
            MORGAN.GetFingerprint(Chem.MolFromSmiles(s))
            for s in ("C", "O", "N", "CC(=O)O", "c1ccccc1O")
        ]
        dist = tanimoto_distance_matrix(fps)
        self.assertTrue(np.all(dist >= 0.0))
        self.assertTrue(np.all(dist <= 1.0 + 1e-15))

        condensed = squareform(dist, checks=True)
        self.assertEqual(len(condensed), condensed_pair_count(len(fps)))


class TestPairwiseMeanMax(unittest.TestCase):
    def test_known_vector(self) -> None:
        mean, mx = pairwise_mean_max([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(mx, 3.0)

    def test_empty_is_nan(self) -> None:
        mean, mx = pairwise_mean_max([])
        self.assertTrue(math.isnan(mean))
        self.assertTrue(math.isnan(mx))

    def test_condensed_pair_count(self) -> None:
        self.assertEqual(condensed_pair_count(30), 435)
        self.assertEqual(condensed_pair_count(1), 0)
        self.assertEqual(condensed_pair_count(2), 1)


class TestEnsembleStats(unittest.TestCase):
    def test_heavy_atom_indices_exclude_h(self) -> None:
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        heavy = heavy_atom_indices(mol)
        self.assertEqual(len(heavy), 3)
        self.assertTrue(all(mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in heavy))

    def test_rmsd_matrix_length_and_heavy_vs_allatom(self) -> None:
        mol = _embed_confs("CCCC", n_confs=5, seed=7)
        n = mol.GetNumConformers()
        heavy = heavy_atom_indices(mol)
        all_mat = np.asarray(AllChem.GetConformerRMSMatrix(mol, prealigned=False))

        mol_h = Chem.Mol(mol)
        mean_h, max_h = ensemble_heavy_rmsd_stats(mol_h)
        heavy_mat = np.asarray(
            AllChem.GetConformerRMSMatrix(
                Chem.Mol(mol), atomIds=heavy, prealigned=False
            )
        )
        self.assertEqual(len(heavy_mat), condensed_pair_count(n))
        self.assertAlmostEqual(mean_h, float(heavy_mat.mean()), places=6)
        self.assertAlmostEqual(max_h, float(heavy_mat.max()), places=6)

        self.assertFalse(np.allclose(heavy_mat, all_mat))
        self.assertLess(float(heavy_mat.mean()), float(all_mat.mean()))

    def test_tfd_matrix_length_and_bounds(self) -> None:
        mol = _embed_confs("CCCC", n_confs=5, seed=3)
        mean_t, max_t = ensemble_tfd_stats(mol)
        self.assertGreaterEqual(mean_t, 0.0)
        self.assertLessEqual(max_t, 1.0 + 1e-9)
        self.assertLessEqual(mean_t, max_t + 1e-12)

    def test_identical_conformers_near_zero_rmsd(self) -> None:
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        conf0 = mol.GetConformer(0)
        for _ in range(3):
            mol.AddConformer(Chem.Conformer(conf0), assignId=True)
        mean_r, max_r = ensemble_heavy_rmsd_stats(mol)
        self.assertLess(mean_r, 1e-4)
        self.assertLess(max_r, 1e-4)


class TestMorganIsEcfp4(unittest.TestCase):
    def test_plot_morgan_matches_radius_two(self) -> None:
        """ECFP4 ≡ Morgan fingerprint with radius 2 (diameter 4)."""
        from plot_dataset_diversity_panels import MORGAN as plot_morgan

        ref = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        fp_a = plot_morgan.GetFingerprint(mol)
        fp_b = ref.GetFingerprint(mol)
        self.assertEqual(DataStructs.TanimotoSimilarity(fp_a, fp_b), 1.0)


if __name__ == "__main__":
    unittest.main()
