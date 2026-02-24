import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T1_mm_1ano"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MM_CSV = DADOS_DIR / "base_diaria_interrupcoes_clima_mm.csv"

# Cores (pedido do prof)
COR_INTERRUPCOES = "red"
COR_TEMPERATURA = "blue"
COR_PRECIP = "navy"  # azul mais forte

JANELAS = [7, 14]


def load_base():
    if not BASE_MM_CSV.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {BASE_MM_CSV}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )
    df = pd.read_csv(BASE_MM_CSV)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def plot_ano(df, ano: int):
    sub = df[df["data"].dt.year == ano].copy()
    if sub.empty:
        print(f"[SKIP] Ano {ano}: sem dados.")
        return

    # 1) Interrupções
    out = GRAFICOS_DIR / f"mm_diario_interrupcoes_{ano}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(sub["data"], sub["interrupcoes"], color=COR_INTERRUPCOES, alpha=0.30, label="Interrupções (diário)")
    for w in JANELAS:
        col = f"interrupcoes_mm_{w}"
        if col in sub.columns:
            plt.plot(sub["data"], sub[col], color=COR_INTERRUPCOES, label=f"Interrupções MM{w}")
    plt.title(f"Interrupções diárias com médias móveis ({ano})")
    plt.xlabel("Data")
    plt.ylabel("Interrupções")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    # 2) Temperatura
    out = GRAFICOS_DIR / f"mm_diario_temperatura_{ano}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(sub["data"], sub["temperatura_media"], color=COR_TEMPERATURA, alpha=0.30, label="Temperatura (diário)")
    for w in JANELAS:
        col = f"temperatura_media_mm_{w}"
        if col in sub.columns:
            plt.plot(sub["data"], sub[col], color=COR_TEMPERATURA, label=f"Temperatura MM{w}")
    plt.title(f"Temperatura média diária com médias móveis ({ano})")
    plt.xlabel("Data")
    plt.ylabel("Temperatura (°C)")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    # 3) Precipitação
    out = GRAFICOS_DIR / f"mm_diario_precipitacao_{ano}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(sub["data"], sub["precipitacao_total_mm"], color=COR_PRECIP, alpha=0.30, label="Precipitação (diário)")
    for w in JANELAS:
        col = f"precipitacao_total_mm_mm_{w}"
        if col in sub.columns:
            plt.plot(sub["data"], sub[col], color=COR_PRECIP, label=f"Precipitação MM{w}")
    plt.title(f"Precipitação diária com médias móveis ({ano})")
    plt.xlabel("Data")
    plt.ylabel("Precipitação (mm)")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[OK] Ano {ano}: 3 gráficos gerados.")


def main():
    print("BASE_MM_CSV =", BASE_MM_CSV)
    df = load_base()
    anos = sorted(df["data"].dt.year.unique().tolist())
    print("Anos encontrados:", anos)
    for ano in anos:
        plot_ano(df, int(ano))
    print("\n[OK] T1 concluído: médias móveis diárias por ano + cores padronizadas.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
