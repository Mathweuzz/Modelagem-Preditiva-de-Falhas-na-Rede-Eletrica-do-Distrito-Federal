"""Baseline ingênuo de persistência para os protocolos oficiais do TCC.

Para cada data-alvo, a previsão é o número de interrupções observado na data de
origem: ``y_hat[t+h] = y[t]``. O script não treina parâmetros e produz métricas
nos mesmos recortes usados pela avaliação principal e pela análise
multi-horizonte.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_DIR = Path(__file__).resolve().parent
FONTE_DIR = MODEL_DIR.parents[1]
DATA_PATH = FONTE_DIR / "data" / "dataset_engenharia_features.csv"
OUTPUT_PATH = FONTE_DIR / "results" / "ml" / "metrics_persistence.csv"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula as quatro métricas reportadas para os demais modelos."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
    }


def evaluate_persistence(
    df: pd.DataFrame,
    scope: str,
    horizon: int,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
) -> dict[str, float | int | str]:
    """Avalia ``interrupcoes[t]`` contra ``interrupcoes[t+h]``."""
    targets = pd.date_range(test_start, test_end, freq="D")
    origins = targets - pd.Timedelta(days=horizon)
    missing = targets.difference(df.index).union(origins.difference(df.index))
    if not missing.empty:
        raise ValueError(f"Datas ausentes no dataset: {missing.min()} a {missing.max()}")

    y_true = df.loc[targets, "interrupcoes"].to_numpy()
    y_pred = df.loc[origins, "interrupcoes"].to_numpy()
    return {
        "Escopo": scope,
        "Modelo": "Persistencia",
        "Horizonte": int(horizon),
        **calculate_metrics(y_true, y_pred),
    }


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        evaluate_persistence(
            df,
            "principal_365",
            1,
            "2024-06-01",
            "2025-05-31",
        )
    ]
    rows.extend(
        evaluate_persistence(
            df,
            "multihorizonte_352",
            horizon,
            "2024-06-14",
            "2025-05-31",
        )
        for horizon in (1, 3, 7, 14)
    )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH, index_col="data", parse_dates=True).sort_index()
    metrics = build_metrics(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"\nMétricas salvas em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
