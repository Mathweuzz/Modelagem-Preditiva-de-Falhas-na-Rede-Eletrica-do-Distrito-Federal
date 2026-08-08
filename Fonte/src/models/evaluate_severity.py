"""Evaluate every h=1 model with severity masks from one canonical y_true."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


MODELS_DIR = Path(__file__).resolve().parent
FONTE_DIR = MODELS_DIR.parents[1]
DATASET_PATH = FONTE_DIR / "data" / "dataset_engenharia_features.csv"
RESULTS_DIR = FONTE_DIR / "results" / "ml"
MODEL_FILES = {
    "XGBoost": "predictions_xgboost.csv",
    "Bi-LSTM": "predictions_lstm_bi.csv",
    "Bi-GRU": "predictions_gru_bi.csv",
}
BANDS = (
    ("Normal (<200)", -np.inf, 200),
    ("Moderada (200-400)", 200, 401),
    ("Severa (>400)", 401, np.inf),
)


def canonical_target(dataset_path: Path = DATASET_PATH) -> pd.Series:
    target = pd.read_csv(dataset_path, usecols=["data", "interrupcoes"], parse_dates=["data"])
    values = target.set_index("data")["interrupcoes"]
    rounded = values.round().astype(int)
    if not np.allclose(values.to_numpy(), rounded.to_numpy(), atol=1e-8):
        raise ValueError("Canonical target is not an integer count series.")
    return rounded


def evaluate_severity(
    target: pd.Series,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    predictions: dict[str, pd.Series] = {}
    common_dates: pd.DatetimeIndex | None = None
    for model, filename in MODEL_FILES.items():
        frame = pd.read_csv(results_dir / filename, parse_dates=["data"]).set_index("data")
        predictions[model] = frame["pred"]
        dates = pd.DatetimeIndex(frame.index)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No common prediction dates were found.")

    y_true = target.reindex(common_dates)
    if y_true.isna().any():
        raise ValueError("Canonical target is missing prediction dates.")

    rows: list[dict[str, object]] = []
    for label, lower, upper in BANDS:
        mask = (y_true >= lower) & (y_true < upper)
        row: dict[str, object] = {
            "Faixa": label,
            "Dias": int(mask.sum()),
            "Percentual": float(mask.mean() * 100.0),
        }
        for model, prediction in predictions.items():
            aligned = prediction.reindex(common_dates)
            row[model] = (
                float(mean_absolute_error(y_true[mask], aligned[mask]))
                if mask.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    output = evaluate_severity(canonical_target())
    output.to_csv(RESULTS_DIR / "metrics_severity.csv", index=False)
    print(output.to_string(index=False))
    print("[OK] Severidade calculada com um unico target canonico inteiro.")
