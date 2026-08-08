"""Canonical temporal aggregation and correlation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLE_LABELS = {
    "temperatura_media": "Temperatura media",
    "precipitacao_total_mm": "Precipitacao total",
    "vento_velocidade_media_ms": "Velocidade media do vento",
    "vento_velocidade_max_ms": "Velocidade maxima do vento",
    "vento_rajada_max_ms": "Rajada maxima",
    "vento_dir_sin": "Direcao do vento (seno)",
    "vento_dir_cos": "Direcao do vento (cosseno)",
    "consumo_total_kwh": "Consumo total",
}

SCALE_FREQUENCIES = {"diario": None, "semanal": "W-MON", "mensal": "MS"}


def load_daily_base(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["data"])
    return frame.set_index("data").sort_index()


def aggregate_daily(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Apply one documented rule per variable at weekly/monthly scales."""
    result = frame.resample(frequency).agg(
        interrupcoes=("interrupcoes", "sum"),
        precipitacao_total_mm=("precipitacao_total_mm", "sum"),
        temperatura_media=("temperatura_media", "mean"),
        vento_velocidade_media_ms=("vento_velocidade_media_ms", "mean"),
        vento_velocidade_max_ms=("vento_velocidade_max_ms", "max"),
        vento_rajada_max_ms=("vento_rajada_max_ms", "max"),
        vento_dir_sin=("vento_dir_sin", "mean"),
        vento_dir_cos=("vento_dir_cos", "mean"),
    )
    norm = np.hypot(result["vento_dir_sin"], result["vento_dir_cos"])
    result["vento_dir_sin"] = result["vento_dir_sin"] / norm
    result["vento_dir_cos"] = result["vento_dir_cos"] / norm
    return result


def frames_by_scale(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: frame.copy() if frequency is None else aggregate_daily(frame, frequency)
        for name, frequency in SCALE_FREQUENCIES.items()
    }


def build_correlation_table(
    daily: pd.DataFrame,
    monthly_consumption_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = frames_by_scale(daily)
    rows: list[dict[str, object]] = []
    variables = list(VARIABLE_LABELS)[:-1]
    for scale, frame in frames.items():
        for variable in variables:
            rows.append(
                {
                    "nivel_temporal": scale,
                    "variavel": variable,
                    "pearson_r": float(frame["interrupcoes"].corr(frame[variable])),
                }
            )

    if monthly_consumption_path and Path(monthly_consumption_path).exists():
        consumption = pd.read_csv(monthly_consumption_path)
        consumption["data_referencia"] = pd.to_datetime(consumption["data_referencia"])
        consumption = consumption.set_index("data_referencia")
        joined = frames["mensal"][["interrupcoes"]].join(
            consumption[["consumo_total_kwh"]], how="inner"
        )
        rows.append(
            {
                "nivel_temporal": "mensal",
                "variavel": "consumo_total_kwh",
                "pearson_r": float(joined["interrupcoes"].corr(joined["consumo_total_kwh"])),
            }
        )
    table = pd.DataFrame(rows)
    table["rotulo"] = table["variavel"].map(VARIABLE_LABELS)
    return table, frames


def plot_correlation_table(table: pd.DataFrame, output_path: Path | str) -> None:
    pivot = table.pivot(index="rotulo", columns="nivel_temporal", values="pearson_r")
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.reindex(order)
    scales = ["diario", "semanal", "mensal"]
    colors = ["#4f9bd9", "#f4a236", "#e0533a"]
    y = np.arange(len(pivot))
    height = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    for offset, scale, color in zip([-height, 0, height], scales, colors):
        values = pivot.get(scale, pd.Series(index=pivot.index, dtype=float))
        ax.barh(y + offset, values, height, label=scale.capitalize(), color=color, edgecolor="black")
        for yi, value in zip(y, values):
            if pd.notna(value):
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    yi + offset,
                    f"{value:.2f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                )
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Correlacao de Pearson (r)")
    ax.set_title("Correlacao com o total diario de interrupcoes por escala temporal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
