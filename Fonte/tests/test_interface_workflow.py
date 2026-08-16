"""Testes de integração da interface e de sua camada de serviço."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


FONTE_DIR = Path(__file__).resolve().parents[1]
INTERFACE_DIR = FONTE_DIR / "interface"
APP_PATH = INTERFACE_DIR / "app.py"
sys.path.insert(0, str(INTERFACE_DIR))

import model_service  # noqa: E402


class InterfaceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = model_service.load_dataset()

    def test_dataset_and_reference_artifacts_are_available(self) -> None:
        self.assertEqual(len(self.df), 3066)
        self.assertTrue(self.df.index.is_monotonic_increasing)
        self.assertFalse(self.df.index.has_duplicates)

        direct_metrics, direct_predictions = model_service.load_direct_results()
        self.assertEqual(set(direct_metrics["Modelo"]), {"XGBoost", "Bi-LSTM", "Bi-GRU"})
        self.assertEqual(set(direct_metrics["n"]), {365})
        self.assertEqual(direct_predictions["data_alvo"].nunique(), 365)

        metrics, predictions = model_service.load_reference_results()
        self.assertEqual(set(metrics["Horizonte"]), set(model_service.SUPPORTED_HORIZONS))
        counts = predictions.groupby(["modelo", "horizonte"]).size()
        self.assertTrue(counts.eq(352).all())

    def test_all_interface_pages_render_without_exceptions(self) -> None:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        pages = (
            "Visão geral",
            "Previsão direta",
            "Treinar modelos",
            "Multi-horizonte",
            "Executar análises",
            "Sobre",
        )
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])

        for page in pages[1:]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(
                list(app.exception),
                [],
                msg=f"A página {page!r} apresentou uma exceção.",
            )

    def test_temporal_validation_rejects_overlap_and_missing_horizon_gap(self) -> None:
        common = {
            "df": self.df,
            "models": ["XGBoost"],
            "train_start": "2023-01-01",
            "train_end": "2024-05-31",
            "test_end": "2024-06-30",
        }
        with self.assertRaisesRegex(ValueError, "não podem se sobrepor"):
            model_service.validate_experiment(
                **common,
                horizons=[1],
                test_start="2024-05-31",
            )
        with self.assertRaisesRegex(ValueError, "não é causal"):
            model_service.validate_experiment(
                **common,
                horizons=[14],
                test_start="2024-06-01",
            )

    def test_temporal_validation_rejects_other_invalid_configurations(self) -> None:
        valid = {
            "df": self.df,
            "models": ["XGBoost"],
            "horizons": [1],
            "train_start": "2023-01-01",
            "train_end": "2023-06-30",
            "test_start": "2023-07-01",
            "test_end": "2023-07-07",
        }
        invalid_cases = (
            ({"models": []}, "pelo menos um modelo"),
            ({"horizons": []}, "pelo menos um horizonte"),
            ({"models": ["Modelo inexistente"]}, "não reconhecidos"),
            ({"horizons": [2]}, "não reconhecidos"),
            ({"train_start": "2023-06-30"}, "início do treinamento"),
            ({"test_end": "2023-06-30"}, "início da avaliação"),
            ({"train_start": "2016-01-01"}, "devem estar entre"),
            ({"train_start": "2023-02-01"}, "pelo menos 180 dias"),
            ({"test_end": "2023-07-06"}, "pelo menos 7 dias"),
        )
        for changes, message in invalid_cases:
            with self.subTest(changes=changes):
                arguments = {**valid, **changes}
                with self.assertRaisesRegex(ValueError, message):
                    model_service.validate_experiment(**arguments)

    def test_interface_metrics_follow_the_project_definitions(self) -> None:
        metrics = model_service.calculate_metrics(
            np.array([10.0, 20.0]),
            np.array([8.0, 24.0]),
        )
        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["MAE"], 3.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(10.0))
        self.assertAlmostEqual(metrics["R2"], 0.6)
        self.assertAlmostEqual(metrics["MAPE"], 20.0)

    def test_short_xgboost_run_is_saved_outside_official_results(self) -> None:
        predictions, metrics = model_service.run_experiment(
            self.df,
            models=["XGBoost"],
            horizons=[1],
            train_start="2023-01-01",
            train_end="2023-06-30",
            test_start="2023-07-01",
            test_end="2023-07-07",
        )
        self.assertEqual(len(predictions), 7)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics.iloc[0]["n"], 7)
        self.assertEqual(predictions["data_alvo"].min(), pd.Timestamp("2023-07-01"))
        self.assertEqual(predictions["data_alvo"].max(), pd.Timestamp("2023-07-07"))

        config = {
            "tipo_previsao": "Direta — próximo dia",
            "modelos": ["XGBoost"],
            "horizontes": [1],
            "treino_inicio": "2023-01-01",
            "treino_fim": "2023-06-30",
            "avaliacao_inicio": "2023-07-01",
            "avaliacao_fim": "2023-07-07",
        }
        with tempfile.TemporaryDirectory() as directory:
            temporary_results = Path(directory) / "interface"
            with patch.object(
                model_service,
                "INTERFACE_RESULTS_DIR",
                temporary_results,
            ):
                output = model_service.save_experiment(predictions, metrics, config)

            self.assertEqual(output.parent, temporary_results)
            self.assertTrue((output / "previsoes.csv").is_file())
            self.assertTrue((output / "metricas.csv").is_file())
            saved_config = json.loads(
                (output / "configuracao.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config, config)

    def test_script_execution_is_restricted_to_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "Script não permitido"):
            model_service.run_allowlisted_script("../../arquivo_arbitrario")


if __name__ == "__main__":
    unittest.main()
