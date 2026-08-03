"""Serviços de dados e treinamento usados pela interface Streamlit.

Este módulo mantém a interface separada da camada científica. Ele reutiliza as
arquiteturas do projeto, aplica separação temporal por data-alvo e devolve
DataFrames prontos para visualização e exportação.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


INTERFACE_DIR = Path(__file__).resolve().parent
FONTE_DIR = INTERFACE_DIR.parent
PROJECT_DIR = FONTE_DIR.parent
MODELS_DIR = FONTE_DIR / "src" / "models"
DATASET_PATH = FONTE_DIR / "data" / "dataset_engenharia_features.csv"
ML_RESULTS_DIR = FONTE_DIR / "results" / "ml"
INTERFACE_RESULTS_DIR = FONTE_DIR / "results" / "interface"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from gru_avancada import AdvancedGRU, train_gru_model  # noqa: E402
from lstm_bidirecional import (  # noqa: E402
    AdvancedLSTM,
    SEED,
    set_seeds,
    train_dl_model,
)
from previsao_multihorizonte import criar_sequencias_diretas  # noqa: E402


MODEL_OPTIONS = ("XGBoost", "Bi-LSTM", "Bi-GRU", "ARIMAX")
SUPPORTED_HORIZONS = (1, 3, 7, 14)
ARIMAX_FEATURES = (
    "interrupcoes",
    "temperatura_media",
    "precipitacao_total_mm",
    "vento_rajada_max_ms",
    "mes_sin",
    "mes_cos",
)

ProgressCallback = Callable[[str], None]


SCRIPT_CATALOG = {
    "eda_basica": {
        "label": "Análise exploratória básica",
        "description": "Gera série temporal, distribuição, evolução anual e gráficos exploratórios.",
        "path": FONTE_DIR / "src" / "04_eda_basica.py",
        "duration": "aproximadamente 1 minuto",
    },
    "correlacoes": {
        "label": "Correlações não lineares",
        "description": "Calcula Spearman, Kendall e correlações cruzadas.",
        "path": FONTE_DIR / "src" / "02_correlacoes_nao_lineares.py",
        "duration": "aproximadamente 1 minuto",
    },
    "engenharia": {
        "label": "Engenharia de atributos",
        "description": "Reconstrói o dataset final a partir da base diária consolidada.",
        "path": FONTE_DIR / "src" / "03_feature_engineering.py",
        "duration": "menos de 1 minuto",
    },
    "xgboost_completo": {
        "label": "XGBoost com busca de hiperparâmetros",
        "description": "Executa o treinamento completo e atualiza métricas, previsões e figuras.",
        "path": MODELS_DIR / "baseline_xgboost.py",
        "duration": "pode levar vários minutos",
    },
    "lstm_completa": {
        "label": "Bi-LSTM completa (150 épocas)",
        "description": "Reproduz o treinamento Bi-LSTM utilizado no trabalho.",
        "path": MODELS_DIR / "lstm_bidirecional.py",
        "duration": "aproximadamente 5–15 minutos",
    },
    "gru_completa": {
        "label": "Bi-GRU completa (150 épocas)",
        "description": "Reproduz o treinamento Bi-GRU utilizado no trabalho.",
        "path": MODELS_DIR / "gru_avancada.py",
        "duration": "aproximadamente 5–15 minutos",
    },
    "multihorizonte_completo": {
        "label": "Comparação multi-horizonte completa",
        "description": "Treina os três modelos para 1, 3, 7 e 14 dias.",
        "path": MODELS_DIR / "previsao_multihorizonte.py",
        "duration": "pode levar de 30 minutos a mais de 1 hora",
    },
    "graficos_multihorizonte": {
        "label": "Atualizar gráficos multi-horizonte",
        "description": "Regenera os gráficos a partir das métricas já calculadas.",
        "path": MODELS_DIR / "plot_multihorizonte.py",
        "duration": "menos de 1 minuto",
    },
}


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Carrega e valida o dataset de engenharia de atributos."""
    df = pd.read_csv(path, index_col="data", parse_dates=True)
    df = df.sort_index()
    if df.empty:
        raise ValueError("O dataset está vazio.")
    if "interrupcoes" not in df.columns:
        raise ValueError("A coluna obrigatória 'interrupcoes' não foi encontrada.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("As datas do dataset não estão em ordem crescente.")
    if df.index.has_duplicates:
        raise ValueError("O dataset contém datas duplicadas.")
    return df


def load_reference_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega métricas e previsões multi-horizonte já reproduzidas."""
    metrics_path = ML_RESULTS_DIR / "metrics_multihorizon.csv"
    predictions_path = ML_RESULTS_DIR / "predictions_all.csv"
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(
        predictions_path,
        parse_dates=["data_origem", "data_alvo"],
    )
    return metrics, predictions


def load_direct_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os artefatos específicos da previsão direta do próximo dia."""
    files = {
        "XGBoost": ("metrics_xgboost.csv", "predictions_xgboost.csv"),
        "Bi-LSTM": ("metrics_lstm_bi.csv", "predictions_lstm_bi.csv"),
        "Bi-GRU": ("metrics_gru_bi.csv", "predictions_gru_bi.csv"),
    }
    metric_rows = []
    prediction_frames = []

    for model_name, (metrics_file, predictions_file) in files.items():
        metrics = pd.read_csv(ML_RESULTS_DIR / metrics_file).iloc[0]
        predictions = pd.read_csv(
            ML_RESULTS_DIR / predictions_file,
            parse_dates=["data"],
        )
        metric_rows.append(
            {
                "Modelo": model_name,
                "Horizonte": 1,
                "n": len(predictions),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE": metrics["MAPE"],
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "modelo": model_name,
                    "horizonte": 1,
                    "data_origem": predictions["data"] - pd.Timedelta(days=1),
                    "data_alvo": predictions["data"],
                    "y_real": predictions["real"],
                    "y_pred": predictions["pred"],
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(
        prediction_frames,
        ignore_index=True,
    )

def _timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def validate_experiment(
    df: pd.DataFrame,
    models: Iterable[str],
    horizons: Iterable[int],
    train_start: date | datetime | str | pd.Timestamp,
    train_end: date | datetime | str | pd.Timestamp,
    test_start: date | datetime | str | pd.Timestamp,
    test_end: date | datetime | str | pd.Timestamp,
) -> None:
    """Valida seleções da interface antes de iniciar um treinamento custoso."""
    models = tuple(models)
    horizons = tuple(int(h) for h in horizons)
    train_start = _timestamp(train_start)
    train_end = _timestamp(train_end)
    test_start = _timestamp(test_start)
    test_end = _timestamp(test_end)

    unknown_models = set(models) - set(MODEL_OPTIONS)
    unknown_horizons = set(horizons) - set(SUPPORTED_HORIZONS)

    if not models:
        raise ValueError("Selecione pelo menos um modelo.")
    if not horizons:
        raise ValueError("Selecione pelo menos um horizonte.")
    if unknown_models:
        raise ValueError(f"Modelos não reconhecidos: {sorted(unknown_models)}")
    if unknown_horizons:
        raise ValueError(f"Horizontes não reconhecidos: {sorted(unknown_horizons)}")
    if train_start >= train_end:
        raise ValueError("O início do treinamento deve ser anterior ao fim.")
    if test_start > test_end:
        raise ValueError("O início da avaliação deve ser anterior ao fim.")
    if train_end >= test_start:
        raise ValueError(
            "Treinamento e avaliação não podem se sobrepor. "
            "A avaliação deve começar depois do fim do treinamento."
        )
    earliest_test_origin = test_start - pd.Timedelta(days=max(horizons))
    if train_end > earliest_test_origin:
        minimum_test_start = train_end + pd.Timedelta(days=max(horizons))
        raise ValueError(
            "O início da avaliação não é causal para o maior horizonte: "
            f"com treino até {train_end.date()} e h={max(horizons)}, use "
            f"avaliação a partir de {minimum_test_start.date()}."
        )
    if train_start < df.index.min() or test_end > df.index.max():
        raise ValueError(
            f"As datas devem estar entre {df.index.min().date()} e "
            f"{df.index.max().date()}."
        )

    training_days = int((train_end - train_start).days) + 1
    evaluation_days = int((test_end - test_start).days) + 1
    if training_days < 180:
        raise ValueError("Use pelo menos 180 dias para treinamento.")
    if evaluation_days < 7:
        raise ValueError("Use pelo menos 7 dias para avaliação.")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula as métricas adotadas no trabalho."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MAPE": float(mape),
    }


def _tabular_split(
    df: pd.DataFrame,
    horizon: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Monta a tarefa direta X[t] -> y[t+h] e separa pela data-alvo."""
    y_shifted = df["interrupcoes"].shift(-horizon)
    valid = y_shifted.notna()
    X = df.loc[valid].copy()
    y = y_shifted.loc[valid]
    target_dates = X.index + pd.Timedelta(days=horizon)

    train_mask = (target_dates >= train_start) & (target_dates <= train_end)
    test_mask = (target_dates >= test_start) & (target_dates <= test_end)
    if not train_mask.any():
        raise ValueError("Nenhuma amostra de treinamento foi encontrada.")
    if not test_mask.any():
        raise ValueError(
            f"Nenhuma data-alvo foi encontrada para o horizonte de {horizon} dia(s)."
        )

    return (
        X.loc[train_mask],
        y.loc[train_mask],
        X.loc[test_mask],
        y.loc[test_mask],
        X.index[train_mask] + pd.Timedelta(days=horizon),
        X.index[test_mask] + pd.Timedelta(days=horizon),
    )


def _xgboost_params() -> dict:
    params_path = ML_RESULTS_DIR / "xgboost_best_params.json"
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    return {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
    }


def _prediction_rows(
    model_name: str,
    horizon: int,
    origins: pd.DatetimeIndex,
    targets: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:
    return [
        {
            "modelo": model_name,
            "horizonte": int(horizon),
            "data_origem": origin,
            "data_alvo": target,
            "y_real": float(real),
            "y_pred": float(max(0, pred)),
        }
        for origin, target, real, pred in zip(
            origins,
            targets,
            y_true,
            y_pred,
        )
    ]


def _run_xgboost(
    df: pd.DataFrame,
    horizon: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> list[dict]:
    X_train, y_train, X_test, y_test, _, test_targets = _tabular_split(
        df,
        horizon,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    model = xgb.XGBRegressor(
        **_xgboost_params(),
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = np.maximum(0, model.predict(X_test))
    return _prediction_rows(
        "XGBoost",
        horizon,
        X_test.index,
        test_targets,
        y_test.to_numpy(),
        predictions,
    )


def _run_deep_learning(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    epochs: int,
    sequence_length: int,
) -> list[dict]:
    set_seeds(SEED)
    scaler_mask = (df.index >= train_start) & (df.index <= train_end)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[scaler_mask])
    scaled = scaler.transform(df)
    target_idx = df.columns.get_loc("interrupcoes")

    X_all, y_all, origins, targets = criar_sequencias_diretas(
        df,
        scaled,
        target_idx,
        sequence_length,
        horizon,
    )
    train_mask = (targets >= train_start) & (targets <= train_end)
    test_mask = (targets >= test_start) & (targets <= test_end)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Não há sequências suficientes para {model_name}, horizonte {horizon}."
        )

    X_train = torch.from_numpy(X_all[train_mask]).float()
    y_train = torch.from_numpy(y_all[train_mask]).float().unsqueeze(1)
    X_test = torch.from_numpy(X_all[test_mask]).float()
    input_size = X_train.shape[2]

    if model_name == "Bi-LSTM":
        model = AdvancedLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.4,
        )
        model, _ = train_dl_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
        )
    else:
        model = AdvancedGRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout_rate=0.4,
        )
        model, _ = train_gru_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        normalized_predictions = (
            model(X_test.to(device)).cpu().numpy().flatten()
        )

    dummy = np.zeros((len(normalized_predictions), df.shape[1]))
    dummy[:, target_idx] = normalized_predictions
    predictions = np.maximum(
        0,
        scaler.inverse_transform(dummy)[:, target_idx],
    )

    normalized_true = y_all[test_mask]
    dummy_true = np.zeros((len(normalized_true), df.shape[1]))
    dummy_true[:, target_idx] = normalized_true
    y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

    del model, X_train, y_train, X_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _prediction_rows(
        model_name,
        horizon,
        origins[test_mask],
        targets[test_mask],
        y_true,
        predictions,
    )


def _run_arimax(
    df: pd.DataFrame,
    horizon: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> list[dict]:
    """Executa um ARIMAX direto (1,0,1) com covariáveis conhecidas na origem."""
    X_train, y_train, X_test, y_test, train_targets, test_targets = _tabular_split(
        df,
        horizon,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    features = [column for column in ARIMAX_FEATURES if column in X_train.columns]
    standardizer = StandardScaler()
    exog_train = standardizer.fit_transform(X_train[features])
    endog = pd.Series(y_train.to_numpy(dtype=float), index=train_targets)
    exog_train_df = pd.DataFrame(
        exog_train,
        index=train_targets,
        columns=features,
    )
    # O componente autorregressivo precisa avançar por todas as datas posteriores
    # ao último alvo de treino. Quando a avaliação começa após uma lacuna causal
    # (por exemplo, h=14), prever apenas ``len(test_targets)`` passos faria o
    # Statsmodels gerar valores para datas anteriores e estes seriam rotulados
    # incorretamente com as datas de teste. Construímos, portanto, as covariáveis
    # de todos os alvos intermediários e selecionamos o recorte solicitado somente
    # depois de atualizar o estado do ARIMAX ao longo da lacuna.
    forecast_targets = pd.date_range(
        train_targets.max() + pd.Timedelta(days=1),
        test_targets.max(),
        freq="D",
    )
    forecast_origins = forecast_targets - pd.Timedelta(days=horizon)
    missing_origins = forecast_origins.difference(df.index)
    if not missing_origins.empty:
        raise ValueError(
            "Faltam datas de origem para avançar o ARIMAX até o período de teste: "
            f"{missing_origins.min().date()} a {missing_origins.max().date()}."
        )
    exog_forecast = standardizer.transform(df.loc[forecast_origins, features])
    exog_forecast_df = pd.DataFrame(
        exog_forecast,
        index=forecast_targets,
        columns=features,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            endog=endog,
            exog=exog_train_df,
            order=(1, 0, 1),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)
        forecast = fitted.get_forecast(
            steps=len(exog_forecast_df),
            exog=exog_forecast_df,
        ).predicted_mean
        predictions = forecast.loc[test_targets].to_numpy()

    return _prediction_rows(
        "ARIMAX",
        horizon,
        X_test.index,
        test_targets,
        y_test.to_numpy(),
        np.maximum(0, predictions),
    )


def run_experiment(
    df: pd.DataFrame,
    models: Iterable[str],
    horizons: Iterable[int],
    train_start: date | datetime | str | pd.Timestamp,
    train_end: date | datetime | str | pd.Timestamp,
    test_start: date | datetime | str | pd.Timestamp,
    test_end: date | datetime | str | pd.Timestamp,
    epochs: int = 30,
    sequence_length: int = 14,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina as combinações selecionadas e retorna previsões e métricas."""
    models = tuple(models)
    horizons = tuple(int(h) for h in horizons)
    validate_experiment(
        df,
        models,
        horizons,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    train_start = _timestamp(train_start)
    train_end = _timestamp(train_end)
    test_start = _timestamp(test_start)
    test_end = _timestamp(test_end)
    progress = progress or (lambda _message: None)

    all_rows: list[dict] = []
    total = len(models) * len(horizons)
    current = 0
    for model_name in models:
        for horizon in horizons:
            current += 1
            progress(
                f"[{current}/{total}] Treinando {model_name} para "
                f"{horizon} dia(s)..."
            )
            if model_name == "XGBoost":
                rows = _run_xgboost(
                    df,
                    horizon,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                )
            elif model_name in {"Bi-LSTM", "Bi-GRU"}:
                rows = _run_deep_learning(
                    df,
                    model_name,
                    horizon,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    int(epochs),
                    int(sequence_length),
                )
            else:
                rows = _run_arimax(
                    df,
                    horizon,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                )
            all_rows.extend(rows)

    predictions = pd.DataFrame(all_rows).sort_values(
        ["horizonte", "modelo", "data_alvo"]
    )
    metric_rows = []
    for (model_name, horizon), group in predictions.groupby(
        ["modelo", "horizonte"],
        sort=True,
    ):
        metric_rows.append(
            {
                "Modelo": model_name,
                "Horizonte": int(horizon),
                **calculate_metrics(
                    group["y_real"].to_numpy(),
                    group["y_pred"].to_numpy(),
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["Horizonte", "MAE"])
    progress("Treinamento concluído.")
    return predictions.reset_index(drop=True), metrics.reset_index(drop=True)


def save_experiment(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict,
) -> Path:
    """Persiste cada execução em uma pasta própria, sem sobrescrever resultados."""
    run_name = datetime.now().strftime("execucao_%Y%m%d_%H%M%S_%f")
    output_dir = INTERFACE_RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "previsoes.csv", index=False)
    metrics.to_csv(output_dir / "metricas.csv", index=False)
    (output_dir / "configuracao.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return output_dir


def run_allowlisted_script(script_key: str, timeout_seconds: int = 7200) -> str:
    """Executa somente scripts previamente cadastrados e captura seu relatório."""
    if script_key not in SCRIPT_CATALOG:
        raise ValueError("Script não permitido.")
    script_path = Path(SCRIPT_CATALOG[script_key]["path"]).resolve()
    if not script_path.is_file() or FONTE_DIR.resolve() not in script_path.parents:
        raise FileNotFoundError(f"Script não encontrado: {script_path}")

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=environment,
        check=False,
    )
    report = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"O script terminou com código {completed.returncode}.\n\n{report}"
        )
    return report or "Script concluído sem mensagens de saída."
