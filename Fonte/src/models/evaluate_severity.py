"""Avalia os modelos de h=1 com faixas criadas sobre um único alvo canônico."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = MODELS_DIR.parent
FONTE_DIR = SOURCE_DIR.parent
sys.path.insert(0, str(SOURCE_DIR))

from severity import volume_band_masks  # noqa: E402


DATASET_PATH = FONTE_DIR / "data" / "dataset_engenharia_features.csv"
RESULTS_DIR = FONTE_DIR / "results" / "ml"
MODEL_FILES = {
    "XGBoost": "predictions_xgboost.csv",
    "Bi-LSTM": "predictions_lstm_bi.csv",
    "Bi-GRU": "predictions_gru_bi.csv",
}


def canonical_target(dataset_path: Path | str = DATASET_PATH) -> pd.Series:
    frame = pd.read_csv(
        dataset_path,
        usecols=["data", "interrupcoes"],
        parse_dates=["data"],
    )
    if frame["data"].duplicated().any():
        raise ValueError("O alvo canônico contém datas duplicadas.")
    values = frame.set_index("data")["interrupcoes"].sort_index()
    rounded = values.round().astype(int)
    if not np.allclose(values.to_numpy(), rounded.to_numpy(), atol=1e-8):
        raise ValueError("O alvo canônico deve conter contagens inteiras.")
    return rounded


def _load_predictions(
    results_dir: Path,
    model_files: dict[str, str],
) -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    predictions: dict[str, pd.Series] = {}
    reference_dates: pd.DatetimeIndex | None = None
    for model, filename in model_files.items():
        frame = pd.read_csv(results_dir / filename, parse_dates=["data"])
        if frame["data"].duplicated().any():
            raise ValueError(f"{model} contém datas duplicadas.")
        series = frame.set_index("data")["pred"].sort_index()
        if series.isna().any():
            raise ValueError(f"{model} contém previsões ausentes.")
        dates = pd.DatetimeIndex(series.index)
        if reference_dates is None:
            reference_dates = dates
        elif not dates.equals(reference_dates):
            raise ValueError("Os modelos não possuem exatamente as mesmas datas.")
        predictions[model] = series
    if reference_dates is None or reference_dates.empty:
        raise ValueError("Nenhuma previsão foi encontrada.")
    return predictions, reference_dates


def evaluate_severity(
    target: pd.Series,
    results_dir: Path | str = RESULTS_DIR,
    model_files: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Calcula contagens e MAE reutilizando as mesmas máscaras em todos os modelos."""
    selected_files = MODEL_FILES if model_files is None else model_files
    predictions, dates = _load_predictions(Path(results_dir), selected_files)
    y_true = target.reindex(dates)
    if y_true.isna().any():
        raise ValueError("O alvo canônico não cobre todas as datas de previsão.")

    rows: list[dict[str, object]] = []
    for label, mask in volume_band_masks(y_true).items():
        row: dict[str, object] = {
            "Faixa": label,
            "Dias": int(mask.sum()),
            "Percentual": float(mask.mean() * 100),
        }
        for model, prediction in predictions.items():
            row[model] = float((y_true[mask] - prediction[mask]).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "metrics_severity.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = evaluate_severity(
        canonical_target(arguments.dataset),
        arguments.results_dir,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(arguments.output, index=False)
    print(output.to_string(index=False))
    print("[OK] Faixas calculadas sobre um único alvo canônico inteiro.")
