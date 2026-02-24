#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS = ROOT / "dados"
GRAF = ROOT / "graficos" / "T13_comparacao"
GRAF.mkdir(parents=True, exist_ok=True)

BASELINES = DADOS / "previsoes_diarias_baselines.csv"
DL = DADOS / "previsoes_dl_lstm_gru.csv"

OUT = GRAF / "comparacao_previsoes_zoom_1ano.png"


def main():
    if not BASELINES.exists():
        raise FileNotFoundError(f"Faltou: {BASELINES}")
    if not DL.exists():
        raise FileNotFoundError(f"Faltou: {DL}")

    b = pd.read_csv(BASELINES)
    d = pd.read_csv(DL)

    b["data"] = pd.to_datetime(b["data"])
    d["data"] = pd.to_datetime(d["data"])

    # usar somente teste
    b = b[b["conjunto"] == "teste"].copy()
    d = d[d["conjunto"] == "teste"].copy()

    # merge por data
    m = b.merge(
        d[["data", "y_true", "y_pred_lstm", "y_pred_gru"]],
        on="data",
        how="inner"
    )

    # janela último 1 ano
    end = m["data"].max()
    start = end - pd.Timedelta(days=365)
    z = m[(m["data"] >= start) & (m["data"] <= end)].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(z["data"], z["y_true"], label="Real (y)", linewidth=2)
    plt.plot(z["data"], z["prev_persistencia"], label="Persistência (y[t-1])", alpha=0.9)
    plt.plot(z["data"], z["prev_mm7"], label="Média móvel 7 dias", alpha=0.9)
    plt.plot(z["data"], z["y_pred_lstm"], label="LSTM", alpha=0.9)
    plt.plot(z["data"], z["y_pred_gru"], label="GRU", alpha=0.9)

    plt.title("Comparação de previsões (teste) — zoom ~1 ano")
    plt.xlabel("Data")
    plt.ylabel("Interrupções")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    plt.close()

    print("[OK] Gerado:", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
