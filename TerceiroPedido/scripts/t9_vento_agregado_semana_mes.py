#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T9_vento_agregados"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIARIA_VENTO = DADOS_DIR / "base_diaria_interrupcoes_clima_vento.csv"

OUT_SEMANAL = DADOS_DIR / "aggregados_semanal_interrupcoes_vento.csv"
OUT_MENSAL  = DADOS_DIR / "aggregados_mensal_interrupcoes_vento.csv"

# Cores
COR_INTERRUPCOES = "red"
COR_VENTO = "blue"
COR_RAJADA = "navy"
COR_DIRECAO = "purple"


def moda_series(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return float("nan")
    return float(s2.round(0).mode().iloc[0])


def carregar_base():
    if not BASE_DIARIA_VENTO.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {BASE_DIARIA_VENTO}\n"
            "Rode o T8 para gerar base_diaria_interrupcoes_clima_vento.csv."
        )
    df = pd.read_csv(BASE_DIARIA_VENTO)
    if "data" not in df.columns:
        raise ValueError(f"Coluna 'data' não encontrada. Colunas: {list(df.columns)}")
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    need = [
        "interrupcoes",
        "vento_velocidade_media_ms",
        "vento_velocidade_max_ms",
        "vento_rajada_max_ms",
        "vento_direcao_media_gr",
        "vento_direcao_moda_gr",
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada. Colunas: {list(df.columns)}")

    return df


def agregar(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # freq: "W-MON" (semanal) ou "MS" (mensal)
    tmp = df.set_index("data")

    agg = tmp.resample(freq).agg(
        interrupcoes=("interrupcoes", "sum"),
        vento_velocidade_media_ms=("vento_velocidade_media_ms", "mean"),
        vento_velocidade_max_ms=("vento_velocidade_max_ms", "max"),
        vento_rajada_max_ms=("vento_rajada_max_ms", "max"),
        vento_direcao_media_gr=("vento_direcao_media_gr", "mean"),
        vento_direcao_moda_gr=("vento_direcao_moda_gr", moda_series),
        n_dias=("interrupcoes", "count"),
    ).reset_index()

    agg = agg.rename(columns={"data": "data_referencia"})
    return agg


def plot_duplo(df: pd.DataFrame, y2col: str, titulo: str, out_png: Path, cor2: str):
    sub = df.dropna(subset=[y2col]).copy()
    if sub.empty:
        print(f"[SKIP] Sem dados para {y2col}")
        return

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(sub["data_referencia"], sub["interrupcoes"], color=COR_INTERRUPCOES, marker="o", label="Interrupções")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Interrupções", color=COR_INTERRUPCOES)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(sub["data_referencia"], sub[y2col], color=cor2, marker="s", linestyle="--", label=y2col)
    ax2.set_ylabel(y2col, color=cor2)

    plt.title(titulo)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[OK] Gerado:", out_png)


def main():
    print("BASE_DIARIA_VENTO =", BASE_DIARIA_VENTO)
    df = carregar_base()

    # =========================
    # Semanal (segunda como referência)
    # =========================
    print("[1] Agregando semanal (W-MON)...")
    sem = agregar(df, "W-MON")
    print("[1] Salvando:", OUT_SEMANAL)
    sem.to_csv(OUT_SEMANAL, index=False)

    plot_duplo(
        sem, "vento_velocidade_media_ms",
        "Interrupções semanais x Velocidade média do vento (m/s)",
        GRAFICOS_DIR / "semanal_interrupcoes_vs_vento_vel_media.png",
        COR_VENTO
    )
    plot_duplo(
        sem, "vento_rajada_max_ms",
        "Interrupções semanais x Rajada máxima do vento (m/s)",
        GRAFICOS_DIR / "semanal_interrupcoes_vs_rajada_max.png",
        COR_RAJADA
    )
    plot_duplo(
        sem, "vento_direcao_media_gr",
        "Interrupções semanais x Direção média do vento (graus) *",
        GRAFICOS_DIR / "semanal_interrupcoes_vs_direcao_media.png",
        COR_DIRECAO
    )

    # =========================
    # Mensal
    # =========================
    print("[2] Agregando mensal (MS)...")
    men = agregar(df, "MS")
    print("[2] Salvando:", OUT_MENSAL)
    men.to_csv(OUT_MENSAL, index=False)

    plot_duplo(
        men, "vento_velocidade_media_ms",
        "Interrupções mensais x Velocidade média do vento (m/s)",
        GRAFICOS_DIR / "mensal_interrupcoes_vs_vento_vel_media.png",
        COR_VENTO
    )
    plot_duplo(
        men, "vento_rajada_max_ms",
        "Interrupções mensais x Rajada máxima do vento (m/s)",
        GRAFICOS_DIR / "mensal_interrupcoes_vs_rajada_max.png",
        COR_RAJADA
    )
    plot_duplo(
        men, "vento_direcao_media_gr",
        "Interrupções mensais x Direção média do vento (graus) *",
        GRAFICOS_DIR / "mensal_interrupcoes_vs_direcao_media.png",
        COR_DIRECAO
    )

    print("\n[OK] T9 concluído: agregados semanal/mensal de vento + gráficos.")
    print("(*) Nota: direção do vento é variável circular; média simples é uma aproximação inicial.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
