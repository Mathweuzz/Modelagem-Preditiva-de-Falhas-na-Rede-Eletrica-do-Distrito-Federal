import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T5_scatter_regressao"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DADOS_DIR / "base_mensal_interrupcoes_clima_consumo.csv"

# Cores-base: usaremos colormap para anos/tempo; linha de regressão em preto.
REGRESSAO_COR = "black"


def load_df():
    if not CSV_IN.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {CSV_IN}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )
    df = pd.read_csv(CSV_IN)
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)

    for col in ["interrupcoes", "temperatura_media", "consumo_total_kwh"]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada. Colunas: {list(df.columns)}")

    df["ano"] = df["data_referencia"].dt.year

    # Para evitar notação 1e8 e ficar mais claro: converter kWh -> GWh
    df["consumo_gwh"] = df["consumo_total_kwh"] / 1e6

    return df


def regressao_linear(x: np.ndarray, y: np.ndarray):
    """Retorna coef (a,b) de y = a*x + b e R²."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan

    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    return a, b, r2


def plot_scatter_por_ano(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, titulo: str, out_png: Path):
    plt.figure(figsize=(10, 7))

    anos = sorted(df["ano"].unique().tolist())
    cmap = plt.get_cmap("viridis", len(anos))

    for i, ano in enumerate(anos):
        sub = df[df["ano"] == ano]
        plt.scatter(sub[xcol], sub[ycol], color=cmap(i), label=str(ano))

    # Regressão global (todos os pontos)
    a, b, r2 = regressao_linear(df[xcol].to_numpy(), df[ycol].to_numpy())
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 200)
        ys = a * xs + b
        plt.plot(xs, ys, color=REGRESSAO_COR, linewidth=2, label=f"Regressão linear (R²={r2:.3f})")

    plt.title(titulo + "\n(cada ponto = 1 mês)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Ano", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_scatter_gradiente_tempo(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, titulo: str, out_png: Path):
    plt.figure(figsize=(10, 7))

    # Gradiente temporal (2017 azul -> 2025 vermelho) via "coolwarm"
    t = np.linspace(0, 1, len(df))
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(t)

    plt.scatter(df[xcol], df[ycol], c=colors)

    # Regressão global
    a, b, r2 = regressao_linear(df[xcol].to_numpy(), df[ycol].to_numpy())
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 200)
        ys = a * xs + b
        plt.plot(xs, ys, color=REGRESSAO_COR, linewidth=2, label=f"Regressão linear (R²={r2:.3f})")
        plt.legend(loc="best")

    plt.title(titulo + "\n(cada ponto = 1 mês; cor indica avanço no tempo de 2017→2025)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    print("CSV_IN =", CSV_IN)
    df = load_df()
    print(f"[1] Base mensal carregada com {len(df)} meses: {df['data_referencia'].min().date()} a {df['data_referencia'].max().date()}")

    # A) Consumo x Interrupções
    out1 = GRAFICOS_DIR / "scatter_consumo_vs_interrupcoes_por_ano.png"
    print("[2] Gerando:", out1)
    plot_scatter_por_ano(
        df,
        xcol="consumo_gwh",
        ycol="interrupcoes",
        xlabel="Consumo mensal (GWh) — Neoenergia Brasília (Energia Injetada Total)",
        ylabel="Interrupções no mês",
        titulo="Consumo mensal x Interrupções mensais",
        out_png=out1,
    )

    out1b = GRAFICOS_DIR / "scatter_consumo_vs_interrupcoes_gradiente_tempo.png"
    print("[3] Gerando:", out1b)
    plot_scatter_gradiente_tempo(
        df,
        xcol="consumo_gwh",
        ycol="interrupcoes",
        xlabel="Consumo mensal (GWh) — Neoenergia Brasília (Energia Injetada Total)",
        ylabel="Interrupções no mês",
        titulo="Consumo mensal x Interrupções mensais",
        out_png=out1b,
    )

    # B) Temperatura x Consumo
    out2 = GRAFICOS_DIR / "scatter_temperatura_vs_consumo_por_ano.png"
    print("[4] Gerando:", out2)
    plot_scatter_por_ano(
        df,
        xcol="temperatura_media",
        ycol="consumo_gwh",
        xlabel="Temperatura média mensal (°C)",
        ylabel="Consumo mensal (GWh) — Neoenergia Brasília (Energia Injetada Total)",
        titulo="Temperatura média mensal x Consumo mensal",
        out_png=out2,
    )

    out2b = GRAFICOS_DIR / "scatter_temperatura_vs_consumo_gradiente_tempo.png"
    print("[5] Gerando:", out2b)
    plot_scatter_gradiente_tempo(
        df,
        xcol="temperatura_media",
        ycol="consumo_gwh",
        xlabel="Temperatura média mensal (°C)",
        ylabel="Consumo mensal (GWh) — Neoenergia Brasília (Energia Injetada Total)",
        titulo="Temperatura média mensal x Consumo mensal",
        out_png=out2b,
    )

    print("\n[OK] T5 concluído: scatters mensais com cor por ano/gradiente + regressão + R² (e consumo em GWh).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
