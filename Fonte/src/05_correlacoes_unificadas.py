"""Gera agregações, correlações canônicas e a figura comparativa."""

from __future__ import annotations

import argparse
from pathlib import Path

from aggregation import build_correlation_table, load_daily_base, plot_correlation_table


SOURCE_DIR = Path(__file__).resolve().parent
FONTE_DIR = SOURCE_DIR.parent
DEFAULT_DATA_DIR = FONTE_DIR / "data"
DEFAULT_RESULTS_DIR = FONTE_DIR / "results" / "eda"


def generate(data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    daily = load_daily_base(data_dir / "base_diaria_interrupcoes_clima_vento.csv")
    table, frames = build_correlation_table(
        daily,
        data_dir / "base_mensal_interrupcoes_clima_consumo.csv",
    )
    table.to_csv(data_dir / "correlacoes_consolidadas.csv", index=False)
    frames["semanal"].rename_axis("data_referencia").to_csv(
        data_dir / "agregados_semanais_canonicos.csv"
    )
    frames["mensal"].rename_axis("data_referencia").to_csv(
        data_dir / "agregados_mensais_canonicos.csv"
    )
    plot_correlation_table(table, results_dir / "correlacoes_escala_temporal.png")
    print("[OK] Agregações, correlações e figura canônicas geradas.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments.data_dir, arguments.results_dir)
