"""Generate all aggregate datasets, the canonical correlation CSV and its figure."""

from pathlib import Path

from aggregation import build_correlation_table, load_daily_base, plot_correlation_table


SOURCE_DIR = Path(__file__).resolve().parent
FONTE_DIR = SOURCE_DIR.parent
DATA_DIR = FONTE_DIR / "data"
RESULTS_DIR = FONTE_DIR / "results" / "eda"


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily_base(DATA_DIR / "base_diaria_interrupcoes_clima_vento.csv")
    table, frames = build_correlation_table(
        daily,
        DATA_DIR / "base_mensal_interrupcoes_clima_consumo.csv",
    )
    table.to_csv(DATA_DIR / "correlacoes_consolidadas.csv", index=False)
    frames["semanal"].rename_axis("data_referencia").to_csv(
        DATA_DIR / "agregados_semanais_canonicos.csv"
    )
    frames["mensal"].rename_axis("data_referencia").to_csv(
        DATA_DIR / "agregados_mensais_canonicos.csv"
    )
    plot_correlation_table(table, RESULTS_DIR / "correlacoes_escala_temporal.png")
    print("[OK] Correlacoes, tabela e figura geradas pela implementacao canonica.")
