"""
Comparacao temporal e mensal das previsoes multi-horizonte.

Usa as previsoes oficiais ja reproduzidas em ``predictions_all.csv`` para:
  - gerar um painel Real x Previsto por modelo nos horizontes 1, 3, 7 e 14;
  - calcular MAE e RMSE por mes, modelo e horizonte;
  - gerar mapas de calor do MAE mensal.

Execucao:
  cd Fonte/src/models && python plot_multihorizonte_temporal.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODELS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODELS_DIR.parent.parent / "results" / "ml"
PREDICTIONS_CSV = RESULTS_DIR / "predictions_all.csv"
MONTHLY_METRICS_CSV = RESULTS_DIR / "metrics_multihorizon_monthly.csv"

MODELS = ["XGBoost", "Bi-LSTM", "Bi-GRU"]
HORIZONS = [1, 3, 7, 14]
MODEL_SLUGS = {
    "XGBoost": "xgboost",
    "Bi-LSTM": "bilstm",
    "Bi-GRU": "bigru",
}
HORIZON_COLORS = {
    1: "#0072B2",
    3: "#009E73",
    7: "#E69F00",
    14: "#D55E00",
}
MONTH_LABELS = [
    "jun/24", "jul/24", "ago/24", "set/24", "out/24", "nov/24",
    "dez/24", "jan/25", "fev/25", "mar/25", "abr/25", "mai/25",
]


def load_and_validate_predictions() -> pd.DataFrame:
    """Carrega e valida a cobertura comum de 365 datas por combinacao."""
    df = pd.read_csv(
        PREDICTIONS_CSV,
        parse_dates=["data_origem", "data_alvo"],
    )

    required = {
        "modelo", "horizonte", "data_origem", "data_alvo", "y_real", "y_pred"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {PREDICTIONS_CSV}: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Ha valores ausentes nas previsoes multi-horizonte.")
    if df.duplicated(["modelo", "horizonte", "data_alvo"]).any():
        raise ValueError("Ha previsoes duplicadas por modelo, horizonte e data-alvo.")

    expected = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    observed = set(zip(df["modelo"], df["horizonte"]))
    if observed != expected:
        raise ValueError(
            "Combinacoes modelo-horizonte divergentes: "
            f"faltantes={sorted(expected - observed)}, extras={sorted(observed - expected)}"
        )

    reference_dates = None
    for (model, horizon), group in df.groupby(["modelo", "horizonte"]):
        dates = tuple(group.sort_values("data_alvo")["data_alvo"])
        if len(dates) != 365:
            raise ValueError(f"{model}, h={horizon}: esperado n=365, obtido n={len(dates)}")
        if reference_dates is None:
            reference_dates = dates
        elif dates != reference_dates:
            raise ValueError(f"Datas-alvo nao coincidem para {model}, h={horizon}.")

    return df.sort_values(["modelo", "horizonte", "data_alvo"]).reset_index(drop=True)


def calculate_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas em cada mes civil do periodo de teste."""
    work = df.copy()
    work["mes"] = work["data_alvo"].dt.to_period("M").astype(str)

    rows = []
    for (model, horizon, month), group in work.groupby(
        ["modelo", "horizonte", "mes"], sort=False
    ):
        error = group["y_real"] - group["y_pred"]
        rows.append(
            {
                "Modelo": model,
                "Horizonte": int(horizon),
                "Mes": month,
                "n": int(len(group)),
                "MAE": mean_absolute_error(group["y_real"], group["y_pred"]),
                "RMSE": np.sqrt(mean_squared_error(group["y_real"], group["y_pred"])),
                "Erro_Medio": error.mean(),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["Modelo", "Horizonte", "Mes"])
    metrics.to_csv(MONTHLY_METRICS_CSV, index=False)
    return metrics


def shade_alternating_months(ax: plt.Axes, dates: pd.Series) -> None:
    """Marca meses alternados para facilitar a leitura do periodo anual."""
    starts = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="MS")
    if starts[0] > dates.min():
        starts = starts.insert(0, dates.min().normalize())
    ends = list(starts[1:]) + [dates.max().normalize() + pd.Timedelta(days=1)]
    for index, (start, end) in enumerate(zip(starts, ends)):
        if index % 2 == 1:
            ax.axvspan(start, end, color="#ECEFF1", alpha=0.55, zorder=0)


def plot_temporal_panels(df: pd.DataFrame) -> list[Path]:
    """Gera uma figura de quatro horizontes para cada modelo."""
    outputs = []
    sns.set_theme(style="whitegrid")

    for model in MODELS:
        # Proporcao vertical para insercao em pagina A4 retrato na monografia.
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, sharey=True)
        for ax, horizon in zip(axes, HORIZONS):
            group = df.loc[
                df["modelo"].eq(model) & df["horizonte"].eq(horizon)
            ].sort_values("data_alvo")
            shade_alternating_months(ax, group["data_alvo"])
            ax.plot(
                group["data_alvo"], group["y_real"],
                color="#263238", linewidth=1.25, alpha=0.82, label="Real", zorder=2,
            )
            ax.plot(
                group["data_alvo"], group["y_pred"],
                color=HORIZON_COLORS[horizon], linewidth=1.15, alpha=0.92,
                label=f"Previsto (h={horizon})", zorder=3,
            )
            mae = mean_absolute_error(group["y_real"], group["y_pred"])
            ax.set_title(
                f"Horizonte h={horizon} dia(s) — MAE anual = {mae:.2f}",
                loc="left",
                fontsize=12,
            )
            ax.set_ylabel("Interrupções")
            ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", alpha=0.12)

        axes[-1].set_xlabel("Data-alvo do conjunto de teste")
        month_ticks = pd.date_range("2024-06-01", "2025-05-01", freq="MS")
        axes[-1].set_xticks(month_ticks)
        axes[-1].set_xticklabels(MONTH_LABELS)
        axes[-1].set_xlim(pd.Timestamp("2024-06-01"), pd.Timestamp("2025-05-31"))
        fig.suptitle(
            f"{model}: valores reais e previstos ao longo de 12 meses, por horizonte",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        output = RESULTS_DIR / f"previsao_multihorizonte_temporal_{MODEL_SLUGS[model]}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    return outputs


def plot_monthly_mae_heatmaps(metrics: pd.DataFrame) -> Path:
    """Gera mapas de calor mes x horizonte para os tres modelos."""
    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    for ax, model in zip(axes, MODELS):
        model_metrics = metrics.loc[metrics["Modelo"].eq(model)]
        heatmap = model_metrics.pivot(index="Horizonte", columns="Mes", values="MAE")
        heatmap = heatmap.reindex(index=HORIZONS)
        sns.heatmap(
            heatmap,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=20,
            vmax=220,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "MAE"},
        )
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_ylabel("Horizonte (dias)")
        ax.set_xlabel("")
        ax.set_yticklabels([str(h) for h in HORIZONS], rotation=0)

    axes[-1].set_xticklabels(MONTH_LABELS, rotation=0)
    axes[-1].set_xlabel("Mês da data-alvo")
    fig.suptitle(
        "MAE mensal por modelo e horizonte no período de teste",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output = RESULTS_DIR / "previsao_multihorizonte_mae_mensal.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_and_validate_predictions()
    metrics = calculate_monthly_metrics(predictions)
    temporal_outputs = plot_temporal_panels(predictions)
    heatmap_output = plot_monthly_mae_heatmaps(metrics)

    print(f"Metricas mensais: {MONTHLY_METRICS_CSV}")
    for output in temporal_outputs:
        print(f"Painel temporal: {output}")
    print(f"Mapa de calor: {heatmap_output}")


if __name__ == "__main__":
    main()
