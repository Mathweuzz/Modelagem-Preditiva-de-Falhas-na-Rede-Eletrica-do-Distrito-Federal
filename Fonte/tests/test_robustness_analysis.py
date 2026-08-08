"""Regressões para bootstrap, alinhamento e ablação de atributos."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


FONTE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = FONTE_DIR / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from robustness_analysis import feature_groups, moving_block_indices  # noqa: E402


class RobustnessAnalysisTests(unittest.TestCase):
    def test_moving_blocks_keep_consecutive_observations(self) -> None:
        indices = moving_block_indices(20, 4, 5, seed=7)
        self.assertEqual(indices.shape, (4, 20))
        self.assertTrue(((np.diff(indices.reshape(4, 4, 5), axis=2) % 20) == 1).all())

    def test_moving_blocks_are_reproducible(self) -> None:
        first = moving_block_indices(17, 3, 4, seed=42)
        second = moving_block_indices(17, 3, 4, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_feature_groups_are_disjoint_and_exhaustive(self) -> None:
        columns = [
            "interrupcoes",
            "interrupcoes_lag_1",
            "mes_sin",
            "temperatura_media",
            "precipitacao_total_mm_ema_14",
        ]
        groups = feature_groups(columns)
        sets = {name: set(values) for name, values in groups.items()}
        self.assertFalse(sets["historico"] & sets["calendario"])
        self.assertFalse(sets["historico"] & sets["clima"])
        self.assertFalse(sets["calendario"] & sets["clima"])
        self.assertEqual(set().union(*sets.values()), set(columns))

    def test_nonpositive_bootstrap_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            moving_block_indices(0, 10, 7)


if __name__ == "__main__":
    unittest.main()
