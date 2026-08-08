"""Análises de robustez para a avaliação preditiva principal (h=1).

Este módulo acrescenta três verificações ao protocolo oficial de 365 dias:

1. intervalos de confiança por block bootstrap circular de sete dias;
2. ablação dos grupos de atributos no XGBoost com hiperparâmetros fixos;
3. diagnóstico de viés e autocorrelação dos resíduos.

O conjunto de teste nunca é usado para selecionar hiperparâmetros. A ablação
reutiliza a configuração escolhida no Grid Search temporal do modelo completo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from baseline_xgboost import load_and_split_data
from metric_utils import mean_absolute_percentage_error


MODEL_DIR = Path(__file__).resolve().parent
FONTE_DIR = MODEL_DIR.parents[1]
DATA_PATH = FONTE_DIR / "data" / "dataset_engenharia_features.csv"
RESULTS_DIR = FONTE_DIR / "results" / "ml"
PARAMS_PATH = RESULTS_DIR / "xgboost_best_params.json"

MODEL_FILES = {
    "XGBoost": "predictions_xgboost.csv",
    "Bi-LSTM": "predictions_lstm_bi.csv",
    "Bi-GRU": "predictions_gru_bi.csv",
}

MODEL_COLORS = {
    "XGBoost": "#E69F00",
    "Bi-LSTM": "#009E73",
    "Bi-GRU": "#0072B2",
    "Persistência": "#7A7A7A",
}

CALENDAR_COLUMNS = {
    "mes",
    "dia_semana",
    "dia_ano",
    "mes_sin",
    "mes_cos",
}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula as métricas oficiais para um vetor de previsões."""
    true = np.asarray(y_true, dtype=float)
    pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred),
    }


def moving_block_indices(
    n_observations: int,
    n_resamples: int,
    block_length: int,
    seed: int = 42,
) -> np.ndarray:
    """Gera índices para block bootstrap circular com tamanho final fixo."""
    if n_observations <= 0 or n_resamples <= 0 or block_length <= 0:
        raise ValueError("Os tamanhos do bootstrap devem ser positivos.")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % n_observations
    return indices.reshape(n_resamples, -1)[:, :n_observations]


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Retorna o intervalo percentil bilateral de 95%."""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Particiona os preditores em histórico do alvo, calendário e clima."""
    history = [
        column
        for column in columns
        if column == "interrupcoes" or column.startswith("interrupcoes_lag_")
    ]
    calendar = [column for column in columns if column in CALENDAR_COLUMNS]
    reserved = set(history) | set(calendar)
    weather = [column for column in columns if column not in reserved]
    if set(history) & set(calendar) or set(history) & set(weather) or set(calendar) & set(weather):
        raise ValueError("Os grupos de atributos devem ser disjuntos.")
    if set(history) | set(calendar) | set(weather) != set(columns):
        raise ValueError("A partição não cobre todos os atributos.")
    return {"historico": history, "calendario": calendar, "clima": weather}


def load_official_predictions() -> pd.DataFrame:
    """Carrega e alinha previsões oficiais e a referência de persistência."""
    aligned: pd.DataFrame | None = None
    for model, filename in MODEL_FILES.items():
        frame = pd.read_csv(RESULTS_DIR / filename, parse_dates=["data"])
        required = {"data", "real", "pred"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Colunas ausentes em {filename}: {required - set(frame.columns)}")
        frame = frame.set_index("data").sort_index()[["real", "pred"]]
        frame = frame.rename(columns={"real": f"real_{model}", "pred": model})
        aligned = frame if aligned is None else aligned.join(frame, how="inner")

    assert aligned is not None
    real_columns = [column for column in aligned if column.startswith("real_")]
    canonical = aligned[real_columns[0]].to_numpy(dtype=float)
    for column in real_columns[1:]:
        if not np.allclose(canonical, aligned[column].to_numpy(dtype=float)):
            raise ValueError("Os modelos não compartilham o mesmo alvo canônico.")
    if len(aligned) != 365:
        raise ValueError(f"Esperadas 365 datas comuns; foram obtidas {len(aligned)}.")

    data = pd.read_csv(DATA_PATH, index_col="data", parse_dates=True).sort_index()
    origins = aligned.index - pd.Timedelta(days=1)
    if not origins.isin(data.index).all():
        raise ValueError("Há datas de origem ausentes para a persistência.")
    aligned["Persistência"] = data.loc[origins, "interrupcoes"].to_numpy(dtype=float)
    aligned["real"] = canonical
    return aligned[["real", *MODEL_FILES, "Persistência"]]


def bootstrap_model_intervals(
    predictions: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Estima ICs de MAE, RMSE e viés para todos os modelos."""
    y_true = predictions["real"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []
    for model in [*MODEL_FILES, "Persistência"]:
        y_pred = predictions[model].to_numpy(dtype=float)
        residual = y_true - y_pred
        absolute = np.abs(residual)
        squared = residual**2
        mae_samples = absolute[indices].mean(axis=1)
        rmse_samples = np.sqrt(squared[indices].mean(axis=1))
        bias_samples = residual[indices].mean(axis=1)
        mae_low, mae_high = percentile_interval(mae_samples)
        rmse_low, rmse_high = percentile_interval(rmse_samples)
        bias_low, bias_high = percentile_interval(bias_samples)
        rows.append(
            {
                "Modelo": model,
                "n": len(y_true),
                "MAE": float(absolute.mean()),
                "MAE_IC95_inf": mae_low,
                "MAE_IC95_sup": mae_high,
                "RMSE": float(np.sqrt(squared.mean())),
                "RMSE_IC95_inf": rmse_low,
                "RMSE_IC95_sup": rmse_high,
                "Vies": float(residual.mean()),
                "Vies_IC95_inf": bias_low,
                "Vies_IC95_sup": bias_high,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_pairwise_differences(
    predictions: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Compara MAEs por diferenças pareadas; valor negativo favorece A."""
    pairs = [
        ("XGBoost", "Bi-LSTM"),
        ("XGBoost", "Bi-GRU"),
        ("Bi-LSTM", "Bi-GRU"),
        ("XGBoost", "Persistência"),
        ("Bi-LSTM", "Persistência"),
        ("Bi-GRU", "Persistência"),
    ]
    y_true = predictions["real"].to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []
    for model_a, model_b in pairs:
        loss_a = np.abs(y_true - predictions[model_a].to_numpy(dtype=float))
        loss_b = np.abs(y_true - predictions[model_b].to_numpy(dtype=float))
        differences = loss_a - loss_b
        samples = differences[indices].mean(axis=1)
        low, high = percentile_interval(samples)
        rows.append(
            {
                "Modelo_A": model_a,
                "Modelo_B": model_b,
                "Delta_MAE_A_menos_B": float(differences.mean()),
                "IC95_inf": low,
                "IC95_sup": high,
                "IC_exclui_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Resume viés, dispersão, ACF residual e testes de Ljung-Box."""
    y_true = predictions["real"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []
    for model in [*MODEL_FILES, "Persistência"]:
        residual = y_true - predictions[model].to_numpy(dtype=float)
        residual_acf = acf(residual, nlags=14, fft=False)
        ljung = acorr_ljungbox(residual, lags=[7, 14], return_df=True)
        rows.append(
            {
                "Modelo": model,
                "n": len(residual),
                "Vies_medio": float(residual.mean()),
                "Vies_mediano": float(np.median(residual)),
                "Desvio_padrao_residuo": float(residual.std(ddof=1)),
                "ACF_lag_1": float(residual_acf[1]),
                "ACF_lag_7": float(residual_acf[7]),
                "ACF_lag_14": float(residual_acf[14]),
                "Ljung_Box_p_7": float(ljung.loc[7, "lb_pvalue"]),
                "Ljung_Box_p_14": float(ljung.loc[14, "lb_pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def run_xgboost_ablation(indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina variantes do XGBoost com grupos de atributos controlados."""
    x_train, x_test, y_train, y_test = load_and_split_data(DATA_PATH)
    groups = feature_groups(x_train.columns.tolist())
    variants = {
        "Completo": x_train.columns.tolist(),
        "Histórico + calendário": groups["historico"] + groups["calendario"],
        "Clima + calendário": groups["clima"] + groups["calendario"],
        "Somente histórico": groups["historico"],
        "Somente calendário": groups["calendario"],
    }
    with PARAMS_PATH.open(encoding="utf-8") as stream:
        selected_params = json.load(stream)

    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_frame = pd.DataFrame({"data": y_test.index, "real": y_test.to_numpy()})
    for variant, columns in variants.items():
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **selected_params,
        )
        model.fit(x_train[columns], y_train)
        prediction = np.maximum(0, model.predict(x_test[columns]))
        metrics = calculate_metrics(y_test.to_numpy(), prediction)
        absolute = np.abs(y_test.to_numpy(dtype=float) - prediction)
        mae_samples = absolute[indices].mean(axis=1)
        low, high = percentile_interval(mae_samples)
        metrics_rows.append(
            {
                "Variante": variant,
                "n_atributos": len(columns),
                **metrics,
                "MAE_IC95_inf": low,
                "MAE_IC95_sup": high,
            }
        )
        prediction_frame[variant] = prediction

    origins = y_test.index - pd.Timedelta(days=1)
    raw = pd.read_csv(DATA_PATH, index_col="data", parse_dates=True).sort_index()
    persistence = raw.loc[origins, "interrupcoes"].to_numpy(dtype=float)
    persistence_metrics = calculate_metrics(y_test.to_numpy(), persistence)
    persistence_absolute = np.abs(y_test.to_numpy(dtype=float) - persistence)
    persistence_samples = persistence_absolute[indices].mean(axis=1)
    low, high = percentile_interval(persistence_samples)
    metrics_rows.append(
        {
            "Variante": "Persistência",
            "n_atributos": 1,
            **persistence_metrics,
            "MAE_IC95_inf": low,
            "MAE_IC95_sup": high,
        }
    )
    prediction_frame["Persistência"] = persistence
    return pd.DataFrame(metrics_rows), prediction_frame


def bootstrap_ablation_differences(
    predictions: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Compara contrastes planejados da ablação; valor negativo favorece A."""
    pairs = [
        ("Histórico + calendário", "Completo"),
        ("Clima + calendário", "Completo"),
        ("Somente histórico", "Completo"),
        ("Histórico + calendário", "Persistência"),
        ("Clima + calendário", "Somente calendário"),
        ("Histórico + calendário", "Somente histórico"),
    ]
    y_true = predictions["real"].to_numpy(dtype=float)
    rows: list[dict[str, float | str | bool]] = []
    for variant_a, variant_b in pairs:
        loss_a = np.abs(y_true - predictions[variant_a].to_numpy(dtype=float))
        loss_b = np.abs(y_true - predictions[variant_b].to_numpy(dtype=float))
        differences = loss_a - loss_b
        samples = differences[indices].mean(axis=1)
        low, high = percentile_interval(samples)
        rows.append(
            {
                "Variante_A": variant_a,
                "Variante_B": variant_b,
                "Delta_MAE_A_menos_B": float(differences.mean()),
                "IC95_inf": low,
                "IC95_sup": high,
                "IC_exclui_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_block_sensitivity(
    official_predictions: pd.DataFrame,
    ablation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Repete contrastes planejados com blocos de 7, 14 e 30 dias."""
    rows: list[pd.DataFrame] = []
    for block_length in (7, 14, 30):
        indices = moving_block_indices(
            len(official_predictions), 10_000, block_length, seed=42
        )
        official = bootstrap_pairwise_differences(official_predictions, indices)
        official.insert(0, "Analise", "modelos_oficiais")
        official.insert(1, "Bloco_dias", block_length)
        official = official.rename(
            columns={"Modelo_A": "Alternativa_A", "Modelo_B": "Alternativa_B"}
        )
        rows.append(official)

        ablation = bootstrap_ablation_differences(ablation_predictions, indices)
        ablation.insert(0, "Analise", "ablacao_xgboost")
        ablation.insert(1, "Bloco_dias", block_length)
        ablation = ablation.rename(
            columns={"Variante_A": "Alternativa_A", "Variante_B": "Alternativa_B"}
        )
        rows.append(ablation)
    return pd.concat(rows, ignore_index=True)


def plot_model_intervals(intervals: pd.DataFrame, output: Path) -> None:
    """Plota MAE e IC95 para os modelos oficiais."""
    labels = intervals["Modelo"].tolist()
    values = intervals["MAE"].to_numpy()
    lower = values - intervals["MAE_IC95_inf"].to_numpy()
    upper = intervals["MAE_IC95_sup"].to_numpy() - values
    colors = [MODEL_COLORS[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, alpha=0.86, width=0.66)
    ax.errorbar(positions, values, yerr=[lower, upper], fmt="none", color="black", capsize=5)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("MAE (interrupções/dia)")
    ax.set_title("MAE no teste com IC95% por block bootstrap semanal")
    ax.set_ylim(0, max(intervals["MAE_IC95_sup"]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, values):
        ax.text(position, value + 1.2, f"{value:.2f}".replace(".", ","), ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(metrics: pd.DataFrame, output: Path) -> None:
    """Plota o desempenho das variantes controladas do XGBoost."""
    ordered = metrics.sort_values("MAE", ascending=True).reset_index(drop=True)
    values = ordered["MAE"].to_numpy()
    lower = values - ordered["MAE_IC95_inf"].to_numpy()
    upper = ordered["MAE_IC95_sup"].to_numpy() - values
    colors = ["#3B82F6" if name == "Completo" else "#94A3B8" for name in ordered["Variante"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    positions = np.arange(len(ordered))
    ax.barh(positions, values, color=colors, alpha=0.9)
    ax.errorbar(values, positions, xerr=[lower, upper], fmt="none", color="black", capsize=4)
    ax.set_yticks(positions, ordered["Variante"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (interrupções/dia)")
    ax.set_title("Ablação de grupos de atributos no XGBoost (h=1)")
    ax.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, values):
        ax.text(
            value - 0.9,
            position - 0.24,
            f"{value:.2f}".replace(".", ","),
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10.5,
            zorder=5,
        )
    ax.set_xlim(0, max(ordered["MAE_IC95_sup"]) * 1.18)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_acf(predictions: pd.DataFrame, output: Path) -> None:
    """Plota ACF dos resíduos dos três modelos e da persistência."""
    y_true = predictions["real"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    bound = 1.96 / np.sqrt(len(y_true))
    for ax, model in zip(axes.flat, [*MODEL_FILES, "Persistência"]):
        residual = y_true - predictions[model].to_numpy(dtype=float)
        values = acf(residual, nlags=14, fft=False)
        lags = np.arange(1, 15)
        ax.axhspan(-bound, bound, color="#CBD5E1", alpha=0.55)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.vlines(lags, 0, values[1:], color=MODEL_COLORS[model], linewidth=1.8)
        ax.scatter(lags, values[1:], color=MODEL_COLORS[model], s=18, zorder=3)
        ax.set_title(model)
        ax.set_xticks([1, 3, 7, 10, 14])
        ax.grid(alpha=0.18)
    fig.supxlabel("Defasagem (dias)")
    fig.supylabel("Autocorrelação residual")
    fig.suptitle("Dependência temporal remanescente nos resíduos", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_official_predictions()
    indices = moving_block_indices(
        n_observations=len(predictions),
        n_resamples=10_000,
        block_length=7,
        seed=42,
    )

    intervals = bootstrap_model_intervals(predictions, indices)
    pairwise = bootstrap_pairwise_differences(predictions, indices)
    diagnostics = residual_diagnostics(predictions)
    ablation, ablation_predictions = run_xgboost_ablation(indices)
    ablation_differences = bootstrap_ablation_differences(ablation_predictions, indices)
    sensitivity = bootstrap_block_sensitivity(predictions, ablation_predictions)

    intervals.to_csv(RESULTS_DIR / "robustness_model_intervals.csv", index=False)
    pairwise.to_csv(RESULTS_DIR / "robustness_pairwise_differences.csv", index=False)
    diagnostics.to_csv(RESULTS_DIR / "residual_diagnostics.csv", index=False)
    ablation.to_csv(RESULTS_DIR / "ablation_xgboost_metrics.csv", index=False)
    ablation_predictions.to_csv(RESULTS_DIR / "ablation_xgboost_predictions.csv", index=False)
    ablation_differences.to_csv(
        RESULTS_DIR / "ablation_xgboost_differences.csv", index=False
    )
    sensitivity.to_csv(RESULTS_DIR / "bootstrap_block_sensitivity.csv", index=False)

    plot_model_intervals(intervals, RESULTS_DIR / "uncertainty_mae_block_bootstrap.png")
    plot_ablation(ablation, RESULTS_DIR / "ablation_xgboost.png")
    plot_residual_acf(predictions, RESULTS_DIR / "residual_acf_models.png")

    print("\nIntervalos por modelo:\n", intervals.to_string(index=False))
    print("\nDiferenças pareadas de MAE:\n", pairwise.to_string(index=False))
    print("\nDiagnóstico residual:\n", diagnostics.to_string(index=False))
    print("\nAblação do XGBoost:\n", ablation.to_string(index=False))
    print("\nContrastes da ablação:\n", ablation_differences.to_string(index=False))
    print(
        "\nSensibilidade ao tamanho do bloco (primeiras linhas):\n",
        sensitivity.head(12).to_string(index=False),
    )
    print(f"\nArtefatos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
