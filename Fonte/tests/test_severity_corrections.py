"""Regressões para as faixas descritivas e o alvo canônico."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


FONTE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = FONTE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODELS_DIR))

from evaluate_severity import canonical_target, evaluate_severity  # noqa: E402
from severity import classify_volume, volume_band_masks  # noqa: E402


class SeverityCorrectionTests(unittest.TestCase):
    def test_boundaries_are_unambiguous(self) -> None:
        target = pd.Series([199, 200, 400, 401])
        masks = volume_band_masks(target)
        self.assertEqual([int(mask.sum()) for mask in masks.values()], [1, 2, 1])
        labels = classify_volume(target).astype(str).tolist()
        self.assertIn("<200", labels[0])
        self.assertIn("200–400", labels[1])
        self.assertIn("200–400", labels[2])
        self.assertIn(">400", labels[3])

    def test_evaluation_ignores_model_specific_real_columns(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        target = pd.Series([199, 200, 400, 401], index=dates)
        model_files = {
            "Modelo A": "a.csv",
            "Modelo B": "b.csv",
            "Modelo C": "c.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for offset, filename in enumerate(model_files.values(), start=1):
                pd.DataFrame(
                    {
                        "data": dates,
                        "real": [999.0] * 4,
                        "pred": target.to_numpy() + offset,
                    }
                ).to_csv(path / filename, index=False)
            result = evaluate_severity(target, path, model_files)
        self.assertEqual(result["Dias"].tolist(), [1, 2, 1])
        self.assertEqual(result["Modelo A"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(result["Modelo B"].tolist(), [2.0, 2.0, 2.0])

    def test_models_must_share_the_same_dates(self) -> None:
        dates = pd.date_range("2025-01-01", periods=2, freq="D")
        target = pd.Series([100, 300], index=dates)
        model_files = {"A": "a.csv", "B": "b.csv"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"data": dates, "pred": [100, 300]}).to_csv(
                path / "a.csv", index=False
            )
            pd.DataFrame({"data": dates + pd.Timedelta(days=1), "pred": [100, 300]}).to_csv(
                path / "b.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "mesmas datas"):
                evaluate_severity(target, path, model_files)

    def test_canonical_target_requires_integer_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            pd.DataFrame(
                {"data": ["2025-01-01"], "interrupcoes": [10.5]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "contagens inteiras"):
                canonical_target(path)


if __name__ == "__main__":
    unittest.main()
