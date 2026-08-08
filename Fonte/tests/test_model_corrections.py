"""Regressões para arquiteturas bidirecionais e MAPE."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


FONTE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = FONTE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"
INTERFACE_DIR = FONTE_DIR / "interface"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(INTERFACE_DIR))

from gru_avancada import AdvancedGRU  # noqa: E402
from lstm_bidirecional import AdvancedLSTM  # noqa: E402
from metric_utils import mean_absolute_percentage_error  # noqa: E402
from model_service import calculate_metrics as calculate_interface_metrics  # noqa: E402


class FakeBiLSTM(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        _: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch = x.size(0)
        output = torch.full((batch, x.size(1), 4), 99.0)
        hidden = torch.zeros(4, batch, 2)
        hidden[-2] = torch.tensor([1.0, 2.0])
        hidden[-1] = torch.tensor([3.0, 4.0])
        return output, (hidden, hidden.clone())


class FakeBiGRU(nn.Module):
    def forward(self, x: torch.Tensor, _: object) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        output = torch.full((batch, x.size(1), 4), 99.0)
        hidden = torch.zeros(4, batch, 2)
        hidden[-2] = torch.tensor([1.0, 2.0])
        hidden[-1] = torch.tensor([3.0, 4.0])
        return output, hidden


def expose_bidirectional_representation(model: nn.Module) -> None:
    model.fc1 = nn.Identity()
    model.relu = nn.Identity()
    model.dropout = nn.Identity()
    model.fc2 = nn.Identity()


class ModelCorrectionTests(unittest.TestCase):
    def test_lstm_uses_final_hidden_state_from_both_directions(self) -> None:
        model = AdvancedLSTM(3, 2, 2, 1)
        model.lstm = FakeBiLSTM()
        expose_bidirectional_representation(model)
        result = model(torch.zeros(1, 5, 3))
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_gru_uses_final_hidden_state_from_both_directions(self) -> None:
        model = AdvancedGRU(3, 2, 2, 1)
        model.gru = FakeBiGRU()
        expose_bidirectional_representation(model)
        result = model(torch.zeros(1, 5, 3))
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_mape_excludes_zero_targets(self) -> None:
        result = mean_absolute_percentage_error([0.0, 100.0], [50.0, 80.0])
        self.assertAlmostEqual(result, 20.0)

    def test_mape_is_undefined_when_all_targets_are_zero(self) -> None:
        result = mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0])
        self.assertTrue(np.isnan(result))

    def test_interface_uses_the_same_mape_definition(self) -> None:
        result = calculate_interface_metrics(
            np.array([0.0, 100.0]),
            np.array([50.0, 80.0]),
        )
        self.assertAlmostEqual(result["MAPE"], 20.0)


if __name__ == "__main__":
    unittest.main()
