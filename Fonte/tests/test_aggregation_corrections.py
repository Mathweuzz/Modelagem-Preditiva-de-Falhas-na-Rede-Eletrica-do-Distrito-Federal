"""Regressões para as regras canônicas de agregação temporal."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from aggregation import aggregate_daily  # noqa: E402


def aggregation_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    count = len(index)
    return pd.DataFrame(
        {
            "interrupcoes": np.arange(1, count + 1),
            "precipitacao_total_mm": np.ones(count),
            "temperatura_media": np.arange(count, dtype=float) + 20,
            "vento_velocidade_media_ms": np.ones(count),
            "vento_velocidade_max_ms": np.arange(count, dtype=float) + 2,
            "vento_rajada_max_ms": np.arange(count, dtype=float) + 5,
            "vento_dir_sin": np.zeros(count),
            "vento_dir_cos": np.ones(count),
        },
        index=index,
    )


class AggregationCorrectionTests(unittest.TestCase):
    def test_canonical_aggregation_rules(self) -> None:
        frame = aggregation_frame(pd.date_range("2025-01-01", periods=2, freq="D"))
        frame["precipitacao_total_mm"] = [4.0, 5.0]
        result = aggregate_daily(frame, "MS").iloc[0]
        self.assertEqual(result["interrupcoes"], 3)
        self.assertEqual(result["precipitacao_total_mm"], 9.0)
        self.assertEqual(result["temperatura_media"], 20.5)
        self.assertEqual(result["vento_velocidade_max_ms"], 3.0)
        self.assertEqual(result["vento_rajada_max_ms"], 6.0)

    def test_week_w_mon_ends_on_monday(self) -> None:
        index = pd.date_range("2025-01-07", periods=8, freq="D")
        result = aggregate_daily(aggregation_frame(index), "W-MON")
        self.assertEqual(
            result.index.tolist(),
            [pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-20")],
        )
        self.assertEqual(result["interrupcoes"].tolist(), [28, 8])

    def test_aggregate_direction_is_unitary_or_missing(self) -> None:
        frame = aggregation_frame(pd.date_range("2025-01-01", periods=2, freq="D"))
        frame["vento_dir_sin"] = [0.0, 0.0]
        frame["vento_dir_cos"] = [1.0, -1.0]
        result = aggregate_daily(frame, "MS").iloc[0]
        self.assertTrue(np.isnan(result["vento_dir_sin"]))
        self.assertTrue(np.isnan(result["vento_dir_cos"]))

    def test_all_missing_precipitation_remains_missing(self) -> None:
        frame = aggregation_frame(pd.date_range("2025-01-01", periods=2, freq="D"))
        frame["precipitacao_total_mm"] = np.nan
        result = aggregate_daily(frame, "MS").iloc[0]
        self.assertTrue(np.isnan(result["precipitacao_total_mm"]))


if __name__ == "__main__":
    unittest.main()
