import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T4_precipitacao"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

CSV_SEMANAL = DADOS_DIR / "aggregados_semanal_interrupcoes_precipitacao.csv"
CSV_MENSAL = DADOS_DIR / "aggregados_mensal_interrupcoes_precipitacao.csv"

# Cores (pedido do prof)
COR_INTERRUPCOES = "red"
COR_PRECIP = "navy"  # azul mais forte


def load_csv(path: Path, cols_needed):
    if not path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {path}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )
    df = pd.read_csv(path)
    if "data_referencia" not in df.columns:
        raise ValueError(f"Coluna 'data_referencia' não encontrada em {path.name}. Colunas: {list(df.columns)}")

    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)

    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada em {path.name}. Colunas: {list(df.columns)}")

    return df


def plot_duplo_eixo(df: pd.DataFrame, titulo: str, out_png: Path):
    plt.figure(figsize=(12, 6))

    x = df["data_referencia"]

    ax1 = plt.gca()
    ax1.plot(x, df["interrupcoes"], color=COR_INTERRUPCOES, marker="o", label="Interrupções")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Interrupções", color=COR_INTERRUPCOES)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, df["precipitacao_total_mm"], color=COR_PRECIP, marker="s", linestyle="--", label="Precipitação (mm)")
    ax2.set_ylabel("Precipitação (mm)", color=COR_PRECIP)

    plt.title(titulo)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    print("CSV_SEMANAL =", CSV_SEMANAL)
    print("CSV_MENSAL  =", CSV_MENSAL)

    # Semanal
    df_w = load_csv(CSV_SEMANAL, cols_needed=["interrupcoes", "precipitacao_total_mm"])
    out_w = GRAFICOS_DIR / "interrupcoes_vs_precipitacao_semanal.png"
    print("[1] Gerando semanal:", out_w)
    plot_duplo_eixo(df_w, "Interrupções semanais x Precipitação semanal (cores padronizadas)", out_w)

    # Mensal
    df_m = load_csv(CSV_MENSAL, cols_needed=["interrupcoes", "precipitacao_total_mm"])
    out_m = GRAFICOS_DIR / "interrupcoes_vs_precipitacao_mensal.png"
    print("[2] Gerando mensal :", out_m)
    plot_duplo_eixo(df_m, "Interrupções mensais x Precipitação mensal (cores padronizadas)", out_m)

    print("\n[OK] T4 concluído: gráficos interrupções x precipitação (semanal e mensal) com cores padronizadas.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
