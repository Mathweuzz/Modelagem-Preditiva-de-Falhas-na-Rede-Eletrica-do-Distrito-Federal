"""Testes da reconstrução meteorológica e da direção circular."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


FONTE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = FONTE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from build_base_from_standardized import (  # noqa: E402
    build_daily_weather,
    circular_mean_degrees,
)


def load_feature_engineering_module():
    path = SRC_DIR / "03_feature_engineering.py"
    spec = importlib.util.spec_from_file_location("feature_engineering", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feature_engineering = load_feature_engineering_module()


def hourly_frame(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "data",
            "hora_utc",
            "precipitacao_total_horario_mm",
            "temperatura_ar_bulbo_seco_c",
            "vento_direcao_horaria_gr",
            "vento_rajada_max_ms",
            "vento_velocidade_horaria_ms",
        ],
    )


class WeatherReconstructionTests(unittest.TestCase):
    def test_circular_mean_wraps_across_north(self) -> None:
        result = circular_mean_degrees(pd.Series([359.0, 1.0]))
        self.assertTrue(result < 1e-8 or abs(result - 360.0) < 1e-8)

    def test_identical_hourly_duplicates_are_removed(self) -> None:
        frame = hourly_frame(
            [
                ("2025-01-01", "0000 UTC", 1.0, 20.0, 359.0, 8.0, 2.0),
                ("2025-01-01", "0000 UTC", 1.0, 20.0, 359.0, 8.0, 2.0),
                ("2025-01-01", "0100 UTC", 2.0, 22.0, 1.0, 10.0, 4.0),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.csv"
            frame.to_csv(path, index=False)
            daily = build_daily_weather([path]).iloc[0]

        self.assertEqual(daily["n_registros"], 2)
        self.assertAlmostEqual(daily["precipitacao_total_mm"], 3.0)
        self.assertAlmostEqual(daily["vento_velocidade_media_ms"], 3.0)
        self.assertAlmostEqual(daily["vento_dir_sin"], 0.0, places=12)
        self.assertAlmostEqual(daily["vento_dir_cos"], 1.0, places=12)

    def test_conflicting_hourly_duplicates_are_rejected(self) -> None:
        frame = hourly_frame(
            [
                ("2025-01-01", "0000 UTC", 1.0, 20.0, 10.0, 8.0, 2.0),
                ("2025-01-01", "0000 UTC", 1.0, 25.0, 10.0, 8.0, 2.0),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicatas conflitantes"):
                build_daily_weather([path])

    def test_opposite_directions_produce_missing_vector(self) -> None:
        frame = hourly_frame(
            [
                ("2025-01-01", "0000 UTC", 0.0, 20.0, 0.0, 8.0, 2.0),
                ("2025-01-01", "0100 UTC", 0.0, 20.0, 180.0, 8.0, 2.0),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.csv"
            frame.to_csv(path, index=False)
            daily = build_daily_weather([path]).iloc[0]

        self.assertTrue(np.isnan(daily["vento_dir_sin"]))
        self.assertTrue(np.isnan(daily["vento_dir_cos"]))

    def test_interpolated_direction_is_renormalized(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            {
                "vento_dir_sin": [0.0, np.nan, 1.0],
                "vento_dir_cos": [1.0, np.nan, 0.0],
            },
            index=index,
        )
        result = feature_engineering.fix_continuity(frame)
        norm = np.hypot(result["vento_dir_sin"], result["vento_dir_cos"])
        np.testing.assert_allclose(norm.to_numpy(), np.ones(3))


if __name__ == "__main__":
    unittest.main()
