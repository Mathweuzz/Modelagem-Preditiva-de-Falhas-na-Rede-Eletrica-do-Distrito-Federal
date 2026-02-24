import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1) CAMINHOS
# =====================================================

THIS_FILE = Path(__file__).resolve()

# /home/mateus/CLEAR DATA
ROOT_DIR = THIS_FILE.parents[3]

# Pasta da nova entrega (inner SegundoPedido)
SEGUNDO_PEDIDO_DIR = THIS_FILE.parents[1]
DADOS_DIR = SEGUNDO_PEDIDO_DIR / "dados"
GRAFICOS_DIR = SEGUNDO_PEDIDO_DIR / "graficos"

DADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# Base mensal completa (interrupções + clima + consumo)
BASE_MENSAL_CSV = DADOS_DIR / "base_mensal_interrupcoes_clima_consumo.csv"


# =====================================================
# 2) FUNÇÕES
# =====================================================

def carregar_base_mensal(caminho: Path) -> pd.DataFrame:
    print(f"[1] Lendo base mensal completa de: {caminho}")
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho)

    esperadas = [
        "data_referencia",
        "temperatura_media",
        "consumo_total_kwh",
    ]
    for col in esperadas:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)

    print(f"[1] Base mensal com {len(df)} meses (de "
          f"{df['data_referencia'].min().date()} até {df['data_referencia'].max().date()}).")

    return df


def plotar_tseries_consumo_temperatura(df: pd.DataFrame, caminho_png: Path):
    print(f"[2] Gerando série temporal consumo x temperatura: {caminho_png}")
    plt.figure(figsize=(12, 6))

    x = df["data_referencia"]

    ax1 = plt.gca()
    ax1.plot(x, df["temperatura_media"], marker="o", label="Temperatura média mensal (°C)")
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Temperatura média (°C)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, df["consumo_total_kwh"], marker="s", linestyle="--", label="Consumo mensal (kWh)")
    ax2.set_ylabel("Consumo mensal (kWh)")

    plt.title("Temperatura média mensal x consumo mensal de energia")

    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300)
    plt.close()


def plotar_scatter_consumo_temperatura(df: pd.DataFrame, caminho_png: Path):
    print(f"[3] Gerando scatter consumo x temperatura: {caminho_png}")
    plt.figure(figsize=(8, 6))

    plt.scatter(df["temperatura_media"], df["consumo_total_kwh"])
    plt.xlabel("Temperatura média mensal (°C)")
    plt.ylabel("Consumo mensal (kWh)")
    plt.title("Relação entre consumo mensal e temperatura média mensal")

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300)
    plt.close()


def calcular_correlacao(df: pd.DataFrame):
    print("[4] Calculando correlação entre consumo e temperatura mensal...")
    corr = df["consumo_total_kwh"].corr(df["temperatura_media"])
    print(f"[4] Correlação de Pearson (consumo_total_kwh x temperatura_media): {corr:.4f}")
    return corr


# =====================================================
# 3) MAIN
# =====================================================

def main():
    print("ROOT_DIR =", ROOT_DIR)
    print("BASE_MENSAL_CSV =", BASE_MENSAL_CSV)
    print("SEGUNDO_PEDIDO_DIR =", SEGUNDO_PEDIDO_DIR)

    df = carregar_base_mensal(BASE_MENSAL_CSV)

    # Série temporal
    tseries_png = GRAFICOS_DIR / "mensal_consumo_vs_temperatura_tseries.png"
    plotar_tseries_consumo_temperatura(df, tseries_png)

    # Scatter
    scatter_png = GRAFICOS_DIR / "mensal_scatter_consumo_vs_temperatura.png"
    plotar_scatter_consumo_temperatura(df, scatter_png)

    # Correlação
    corr = calcular_correlacao(df)

    # Salvar correlação em CSV
    corr_csv = DADOS_DIR / "correlacao_mensal_consumo_temperatura.csv"
    print(f"[5] Salvando correlação em: {corr_csv}")
    pd.DataFrame(
        [{"variavel_x": "temperatura_media",
          "variavel_y": "consumo_total_kwh",
          "correlacao_pearson": corr}]
    ).to_csv(corr_csv, index=False)

    print("\n[OK] Tópico 7 concluído: gráficos e correlação consumo x temperatura (base mensal).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Algo deu errado:")
        print(e)
        sys.exit(1)
