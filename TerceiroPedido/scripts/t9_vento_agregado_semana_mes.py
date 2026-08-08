"""Regenera agregados e figuras de vento com as regras canônicas do projeto."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "Fonte" / "src"
sys.path.insert(0, str(SOURCE_DIR))

from aggregation import aggregate_daily, load_daily_base  # noqa: E402


DEFAULT_BASE = PROJECT_ROOT / "Fonte" / "data" / "base_diaria_interrupcoes_clima_vento.csv"
DEFAULT_DATA_DIR = DELIVERY_ROOT / "dados"
DEFAULT_FIGURE_DIR = DELIVERY_ROOT / "graficos" / "T9_vento_agregados"


def add_display_fields(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    result = aggregate_daily(frame, frequency).copy()
    result["vento_direcao_media_circular_gr"] = (
        np.degrees(np.arctan2(result["vento_dir_sin"], result["vento_dir_cos"]))
        % 360.0
    )
    result["n_dias"] = frame["interrupcoes"].resample(frequency).count()
    return result


def plot_dual_axis(
    frame: pd.DataFrame,
    column: str,
    right_label: str,
    title: str,
    output: Path,
    color: str,
) -> None:
    subset = frame.dropna(subset=[column])
    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_left.plot(
        subset.index, subset["interrupcoes"],
        color="red", marker="o", label="Interrupções",
    )
    ax_left.set_xlabel("Data")
    ax_left.set_ylabel("Interrupções", color="red")
    ax_left.tick_params(axis="x", rotation=45)

    ax_right = ax_left.twinx()
    ax_right.plot(
        subset.index, subset[column],
        color=color, marker="s", linestyle="--", label=right_label,
    )
    ax_right.set_ylabel(right_label, color=color)
    ax_left.set_title(title)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_scale(
    frame: pd.DataFrame,
    frequency: str,
    slug: str,
    label: str,
    data_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    aggregated = add_display_fields(frame, frequency)
    aggregated.rename_axis("data_referencia").to_csv(
        data_dir / f"aggregados_{slug}_interrupcoes_vento.csv"
    )
    plot_dual_axis(
        aggregated,
        "vento_velocidade_media_ms",
        "Velocidade média do vento (m/s)",
        f"Interrupções {label} x Velocidade média do vento (m/s)",
        figure_dir / f"{slug}_interrupcoes_vs_vento_vel_media.png",
        "blue",
    )
    plot_dual_axis(
        aggregated,
        "vento_rajada_max_ms",
        "Rajada máxima (m/s)",
        f"Interrupções {label} x Rajada máxima do vento (m/s)",
        figure_dir / f"{slug}_interrupcoes_vs_rajada_max.png",
        "navy",
    )
    plot_dual_axis(
        aggregated,
        "vento_direcao_media_circular_gr",
        "Direção média circular (°)",
        f"Interrupções {label} x Direção média circular do vento (graus)",
        figure_dir / f"{slug}_interrupcoes_vs_direcao_media.png",
        "purple",
    )
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    frame = load_daily_base(args.base)
    weekly = generate_scale(
        frame, "W-MON", "semanal", "semanais", args.data_dir, args.figure_dir
    )
    monthly = generate_scale(
        frame, "MS", "mensal", "mensais", args.data_dir, args.figure_dir
    )
    print(
        f"[OK] Artefatos agregados regenerados: {len(weekly)} semanas e "
        f"{len(monthly)} meses."
    )


if __name__ == "__main__":
    main()
