import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T6_previsao_zoom_1ano"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DADOS_DIR / "previsoes_diarias_baselines.csv"

# Cores (mantendo padrão visual)
COR_REAL = "black"
COR_PERSIST = "red"
COR_MM7 = "blue"


def load_df():
    if not CSV_IN.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {CSV_IN}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )

    df = pd.read_csv(CSV_IN)

    for col in ["data", "conjunto", "interrupcoes", "prev_persistencia", "prev_mm7"]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada. Colunas: {list(df.columns)}")

    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def plot_ano(df_ano: pd.DataFrame, ano: int, out_png: Path):
    plt.figure(figsize=(12, 6))

    plt.plot(df_ano["data"], df_ano["interrupcoes"], color=COR_REAL, label="Real (y)")
    plt.plot(df_ano["data"], df_ano["prev_persistencia"], color=COR_PERSIST, label="Persistência (y[t-1])", alpha=0.9)
    plt.plot(df_ano["data"], df_ano["prev_mm7"], color=COR_MM7, label="Média móvel 7 dias", alpha=0.9)

    plt.title(f"Previsão de interrupções (teste) — Janela de 1 ano ({ano})")
    plt.xlabel("Data")
    plt.ylabel("Interrupções")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    print("CSV_IN =", CSV_IN)
    df = load_df()

    # Apenas TESTE
    df_test = df[df["conjunto"] == "teste"].copy()
    if df_test.empty:
        raise ValueError("Não há linhas com conjunto == 'teste' no CSV.")

    anos = sorted(df_test["data"].dt.year.unique().tolist())
    print("Anos no teste:", anos)

    for ano in anos:
        sub = df_test[df_test["data"].dt.year == ano].copy()
        if sub.empty:
            continue
        out = GRAFICOS_DIR / f"previsao_baselines_teste_{ano}.png"
        print("[OK] Gerando:", out)
        plot_ano(sub, int(ano), out)

    print("\n[OK] T6 concluído: gráficos de previsão (teste) por ano.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
