"""Testes de regressão para os cortes temporais e métricas do projeto."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


FONTE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = FONTE_DIR / "src" / "models"
INTERFACE_DIR = FONTE_DIR / "interface"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(INTERFACE_DIR))

import baseline_persistence  # noqa: E402
import model_service  # noqa: E402
from data_loader_dl import prepare_data_dl  # noqa: E402
from previsao_multihorizonte import criar_sequencias_diretas  # noqa: E402


class TemporalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = model_service.load_dataset()

    def test_tabular_split_uses_target_dates_and_fixed_horizon(self) -> None:
        horizon = 14
        split = model_service._tabular_split(
            self.df,
            horizon,
            pd.Timestamp("2017-01-22"),
            pd.Timestamp("2024-05-31"),
            pd.Timestamp("2024-06-14"),
            pd.Timestamp("2025-05-31"),
        )
        x_train, _, x_test, _, train_targets, test_targets = split
        self.assertEqual(len(x_test), 352)
        self.assertEqual(test_targets.min(), pd.Timestamp("2024-06-14"))
        self.assertEqual(test_targets.max(), pd.Timestamp("2025-05-31"))
        self.assertTrue(((test_targets - x_test.index).days == horizon).all())
        self.assertLessEqual(train_targets.max(), x_test.index.min())
        self.assertEqual(len(x_train), len(train_targets))

    def test_recurrent_sequences_end_at_origin(self) -> None:
        subset = self.df.iloc[:40]
        scaled = subset.to_numpy(dtype=float)
        target_idx = subset.columns.get_loc("interrupcoes")
        x, y, origins, targets = criar_sequencias_diretas(
            subset,
            scaled,
            target_idx,
            seq_length=14,
            h=7,
        )
        self.assertEqual(x.shape[1:], (14, subset.shape[1]))
        self.assertTrue(((targets - origins).days == 7).all())
        self.assertEqual(x[0, -1, target_idx], subset.iloc[13, target_idx])
        self.assertEqual(y[0], subset.iloc[20, target_idx])

    def test_scaler_is_fitted_only_on_training_partition(self) -> None:
        dates = pd.date_range("2020-01-01", periods=220, freq="D")
        values = np.arange(220, dtype=float)
        frame = pd.DataFrame(
            {"data": dates, "interrupcoes": values, "feature": values * 2}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"
            frame.to_csv(path, index=False)
            *_, scaler, _, test_dates = prepare_data_dl(
                path,
                seq_length=14,
                test_size_days=20,
            )
        self.assertEqual(test_dates.min(), dates[-20])
        self.assertEqual(scaler.data_max_[0], 199.0)
        self.assertEqual(scaler.data_max_[1], 398.0)

    def test_arimax_advances_across_gap_before_selecting_test_dates(self) -> None:
        captured: dict[str, pd.DatetimeIndex] = {}

        class FakeFit:
            def get_forecast(self, steps: int, exog: pd.DataFrame) -> SimpleNamespace:
                captured["index"] = exog.index
                return SimpleNamespace(
                    predicted_mean=pd.Series(np.arange(steps, dtype=float), index=exog.index)
                )

        class FakeSarimax:
            def __init__(self, **_: object) -> None:
                pass

            def fit(self, **_: object) -> FakeFit:
                return FakeFit()

        with patch.object(model_service, "SARIMAX", FakeSarimax):
            rows = model_service._run_arimax(
                self.df,
                14,
                pd.Timestamp("2023-06-01"),
                pd.Timestamp("2024-05-31"),
                pd.Timestamp("2024-06-14"),
                pd.Timestamp("2024-06-20"),
            )

        self.assertEqual(captured["index"].min(), pd.Timestamp("2024-06-01"))
        self.assertEqual(captured["index"].max(), pd.Timestamp("2024-06-20"))
        self.assertEqual(rows[0]["data_alvo"], pd.Timestamp("2024-06-14"))
        self.assertEqual(rows[0]["y_pred"], 13.0)

    def test_persistence_metrics_match_canonical_target_slices(self) -> None:
        metrics = baseline_persistence.build_metrics(self.df)
        direct = metrics.loc[metrics["Escopo"] == "principal_365"].iloc[0]
        self.assertEqual(direct["n"], 365)
        direct_targets = pd.date_range("2024-06-01", "2025-05-31", freq="D")
        direct_true = self.df.loc[direct_targets, "interrupcoes"].to_numpy(dtype=float)
        direct_pred = self.df.loc[
            direct_targets - pd.Timedelta(days=1), "interrupcoes"
        ].to_numpy(dtype=float)
        direct_mae = np.mean(np.abs(direct_true - direct_pred))
        direct_rmse = np.sqrt(np.mean((direct_true - direct_pred) ** 2))
        direct_r2 = 1 - np.sum((direct_true - direct_pred) ** 2) / np.sum(
            (direct_true - direct_true.mean()) ** 2
        )
        self.assertAlmostEqual(direct["MAE"], direct_mae)
        self.assertAlmostEqual(direct["RMSE"], direct_rmse)
        self.assertAlmostEqual(direct["R2"], direct_r2)

        multi = metrics.loc[metrics["Escopo"] == "multihorizonte_352"]
        self.assertEqual(multi["n"].tolist(), [352, 352, 352, 352])
        multi_targets = pd.date_range("2024-06-14", "2025-05-31", freq="D")
        expected_mae = []
        for horizon in (1, 3, 7, 14):
            true = self.df.loc[multi_targets, "interrupcoes"].to_numpy(dtype=float)
            pred = self.df.loc[
                multi_targets - pd.Timedelta(days=horizon), "interrupcoes"
            ].to_numpy(dtype=float)
            expected_mae.append(np.mean(np.abs(true - pred)))
        np.testing.assert_allclose(multi["MAE"].to_numpy(), expected_mae)


if __name__ == "__main__":
    unittest.main()
