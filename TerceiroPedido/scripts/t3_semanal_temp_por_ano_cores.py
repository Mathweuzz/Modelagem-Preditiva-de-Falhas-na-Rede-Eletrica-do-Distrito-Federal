import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T3_semanal_temp_ano"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DADOS_DIR / "aggregados_semanal_interrupcoes_temperatura.csv"

# Cores (pedido do prof)
COR_INTERRUPCOES = "red"
COR_TEMPERATURA = "blue"


def load_df():
    if not CSV_IN.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {CSV_IN}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )
    df = pd.read_csv(CSV_IN)
    if "data_referencia" not in df.columns:
        raise ValueError(f"Coluna 'data_referencia' não encontrada. Colunas: {list(df.columns)}")
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)

    for col in ["interrupcoes", "temperatura_media"]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada. Colunas: {list(df.columns)}")
    return df


def plot_por_ano(df: pd.DataFrame, ano: int):
    sub = df[df["data_referencia"].dt.year == ano].copy()
    if sub.empty:
        print(f"[SKIP] Ano {ano}: sem dados.")
        return

    out = GRAFICOS_DIR / f"semanal_interrupcoes_vs_temperatura_{ano}.png"
    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    ax1.plot(
        sub["data_referencia"], sub["interrupcoes"],
        color=COR_INTERRUPCOES, marker="o", label="Interrupções (semanal)"
    )
    ax1.set_xlabel("Semana")
    ax1.set_ylabel("Interrupções", color=COR_INTERRUPCOES)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        sub["data_referencia"], sub["temperatura_media"],
        color=COR_TEMPERATURA, marker="s", linestyle="--", label="Temperatura média (semanal)"
    )
    ax2.set_ylabel("Temperatura (°C)", color=COR_TEMPERATURA)

    plt.title(f"Interrupções semanais x Temperatura média semanal ({ano})")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[OK] Gerado: {out}")


def main():
    print("CSV_IN =", CSV_IN)
    df = load_df()
    anos = sorted(df["data_referencia"].dt.year.unique().tolist())
    print("Anos encontrados:", anos)

    for ano in anos:
        plot_por_ano(df, int(ano))

    print("\n[OK] T3 concluído: semanal (interrupções x temperatura) por ano + cores.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
