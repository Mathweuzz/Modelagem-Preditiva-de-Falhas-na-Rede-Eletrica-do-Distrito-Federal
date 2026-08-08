"""Regenera os artefatos diários de vento a partir da base canônica A001."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "Fonte" / "data" / "base_diaria_interrupcoes_clima_vento.csv"
DEFAULT_DATA_DIR = DELIVERY_ROOT / "dados"
DEFAULT_FIGURE_DIR = DELIVERY_ROOT / "graficos" / "T8_vento"

WIND_COLUMNS = [
    "vento_velocidade_media_ms",
    "vento_velocidade_max_ms",
    "vento_rajada_max_ms",
    "vento_dir_sin",
    "vento_dir_cos",
]


def load_canonical_base(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["data"]).sort_values("data")
    required = {"data", "interrupcoes", *WIND_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    if frame["data"].duplicated().any():
        raise ValueError("A base canônica contém datas duplicadas.")
    return frame


def save_tables(frame: pd.DataFrame, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    weather_columns = ["data", *WIND_COLUMNS]
    if "n_registros" in frame.columns:
        weather_columns.append("n_registros")
    frame[weather_columns].to_csv(data_dir / "vento_diario_brasilia.csv", index=False)
    frame.to_csv(data_dir / "base_diaria_interrupcoes_clima_vento.csv", index=False)

    rows = [
        {"variavel": column, "pearson_r": frame["interrupcoes"].corr(frame[column])}
        for column in WIND_COLUMNS
    ]
    pd.DataFrame(rows).sort_values("pearson_r", ascending=False).to_csv(
        data_dir / "correlacoes_vento.csv",
        index=False,
    )


def plot_dual_axis(
    frame: pd.DataFrame,
    column: str,
    right_label: str,
    title: str,
    output: Path,
    threshold: float | None = None,
) -> None:
    subset = frame.dropna(subset=[column])
    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_left.plot(
        subset["data"], subset["interrupcoes"],
        color="red", alpha=0.6, label="Interrupções (diário)",
    )
    ax_left.set_xlabel("Data")
    ax_left.set_ylabel("Interrupções", color="red")
    ax_left.tick_params(axis="x", rotation=45)

    ax_right = ax_left.twinx()
    ax_right.plot(
        subset["data"], subset[column],
        color="blue", alpha=0.7, label=right_label,
    )
    if threshold is not None:
        ax_right.axhline(
            threshold,
            color="navy",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"Limiar descritivo ({threshold:g} m/s)",
        )
    ax_right.set_ylabel(right_label, color="blue")
    ax_left.set_title(title)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_canonical_base(args.base)
    save_tables(frame, args.data_dir)
    plot_dual_axis(
        frame,
        "vento_velocidade_media_ms",
        "Velocidade média do vento (m/s)",
        "Interrupções diárias x Velocidade média do vento (m/s)",
        args.figure_dir / "diario_interrupcoes_vs_vento_vel_media.png",
    )
    plot_dual_axis(
        frame,
        "vento_rajada_max_ms",
        "Rajada máxima (m/s)",
        "Interrupções diárias x Rajada máxima do vento (m/s)",
        args.figure_dir / "diario_interrupcoes_vs_rajada_max.png",
        threshold=15.0,
    )
    print(f"[OK] Artefatos diários regenerados a partir de {args.base}")


if __name__ == "__main__":
    main()
