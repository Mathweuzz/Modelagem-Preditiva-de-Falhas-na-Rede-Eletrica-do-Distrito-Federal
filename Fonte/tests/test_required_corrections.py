"""Regression tests for the mandatory V7 corrections."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


FONTE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = FONTE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODELS_DIR))

from aggregation import aggregate_daily  # noqa: E402
from build_base_from_raw import (  # noqa: E402
    build_daily_target,
    circular_mean_degrees,
)
from gru_avancada import AdvancedGRU  # noqa: E402
from lstm_bidirecional import AdvancedLSTM  # noqa: E402


class FakeBiLSTM(nn.Module):
    def forward(self, x: torch.Tensor, _: object) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch = x.size(0)
        out = torch.full((batch, x.size(1), 4), 99.0)
        hidden = torch.zeros(4, batch, 2)
        hidden[-2] = torch.tensor([1.0, 2.0])
        hidden[-1] = torch.tensor([3.0, 4.0])
        return out, (hidden, hidden.clone())


class FakeBiGRU(nn.Module):
    def forward(self, x: torch.Tensor, _: object) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        out = torch.full((batch, x.size(1), 4), 99.0)
        hidden = torch.zeros(4, batch, 2)
        hidden[-2] = torch.tensor([1.0, 2.0])
        hidden[-1] = torch.tensor([3.0, 4.0])
        return out, hidden


def expose_representation(model: nn.Module) -> None:
    model.fc1 = nn.Identity()
    model.relu = nn.Identity()
    model.dropout = nn.Identity()
    model.fc2 = nn.Identity()


class RequiredCorrectionTests(unittest.TestCase):
    def test_lstm_uses_final_hidden_state_from_both_directions(self) -> None:
        model = AdvancedLSTM(3, 2, 2, 1)
        model.lstm = FakeBiLSTM()
        expose_representation(model)
        result = model(torch.zeros(1, 5, 3))
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_gru_uses_final_hidden_state_from_both_directions(self) -> None:
        model = AdvancedGRU(3, 2, 2, 1)
        model.gru = FakeBiGRU()
        expose_representation(model)
        result = model(torch.zeros(1, 5, 3))
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_circular_mean_wraps_across_north(self) -> None:
        result = circular_mean_degrees(pd.Series([359.0, 1.0]))
        self.assertTrue(result < 1e-8 or abs(result - 360.0) < 1e-8)

    def test_daily_target_does_not_filter_causes(self) -> None:
        events = pd.DataFrame(
            {
                "NumOrdemInterrupcao": ["a", "b", "c"],
                "DatInicioInterrupcao": pd.to_datetime(
                    ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-02 03:00"]
                ),
                "DscFatoGeradorInterrupcao": ["MEIO AMBIENTE", "EQUIPAMENTO", "PROGRAMADA"],
            }
        )
        daily, summary = build_daily_target(events)
        self.assertEqual(int(daily["interrupcoes"].sum()), 3)
        self.assertEqual(summary["eventos_unicos"], 3)

    def test_canonical_aggregation_rules(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="D")
        frame = pd.DataFrame(
            {
                "interrupcoes": [2, 3],
                "precipitacao_total_mm": [4.0, 5.0],
                "temperatura_media": [20.0, 24.0],
                "vento_velocidade_media_ms": [1.0, 3.0],
                "vento_velocidade_max_ms": [5.0, 7.0],
                "vento_rajada_max_ms": [8.0, 11.0],
                "vento_dir_sin": [0.0, 0.0],
                "vento_dir_cos": [1.0, 1.0],
            },
            index=index,
        )
        result = aggregate_daily(frame, "MS").iloc[0]
        self.assertEqual(result["interrupcoes"], 5)
        self.assertEqual(result["precipitacao_total_mm"], 9.0)
        self.assertEqual(result["temperatura_media"], 22.0)
        self.assertEqual(result["vento_rajada_max_ms"], 11.0)


if __name__ == "__main__":
    unittest.main()
