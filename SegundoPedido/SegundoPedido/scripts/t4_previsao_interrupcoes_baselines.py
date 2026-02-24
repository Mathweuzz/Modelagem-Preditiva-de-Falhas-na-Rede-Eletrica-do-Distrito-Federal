import sys
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1) CONFIGURAÇÕES DE CAMINHO
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

# Base diária integrada (criada no t3)
BASE_DIARIA_CSV = DADOS_DIR / "base_diaria_interrupcoes_clima.csv"

# Proporção da série usada como treino (restante é teste)
TRAIN_FRACTION = 0.8

# =====================================================
# 2) FUNÇÕES AUXILIARES
# =====================================================

def carregar_base_diaria(caminho_csv: Path) -> pd.DataFrame:
    print(f"[1] Lendo base diária integrada de: {caminho_csv}")
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv)
    if "data" not in df.columns or "interrupcoes" not in df.columns:
        raise ValueError(
            f"Esperado colunas 'data' e 'interrupcoes' em {caminho_csv}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    print(f"[1] Série diária com {len(df)} observações de interrupções.")
    return df[["data", "interrupcoes"]]


def criar_previsoes_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria previsões diárias:
      - baseline_persistencia: previsão de hoje = interrupções de ontem
      - baseline_mm7: previsão = média dos últimos 7 dias (até ontem)
    Importante: para cada dia t, usamos apenas informações até t-1.
    """
    df = df.copy()
    df = df.sort_values("data").reset_index(drop=True)

    # Persistência: y_hat(t) = y(t-1)
    df["prev_persistencia"] = df["interrupcoes"].shift(1)

    # Média móvel de 7 dias (usando apenas dias até t-1)
    # Truque: faz rolling em y, desloca uma linha pro futuro
    mm7 = df["interrupcoes"].rolling(window=7, min_periods=1).mean()
    df["prev_mm7"] = mm7.shift(1)

    # Primeiro dia não tem previsão válida; podemos deixar NaN
    return df


def split_treino_teste(df: pd.DataFrame, train_fraction: float):
    n = len(df)
    n_train = int(math.floor(n * train_fraction))
    if n_train < 2:
        raise ValueError("Poucos dados para dividir em treino e teste.")

    df = df.copy()
    df["conjunto"] = "treino"
    df.loc[df.index >= n_train, "conjunto"] = "teste"

    data_corte = df.loc[n_train, "data"]
    print(f"[2] Divisão temporal: {n_train} dias (treino) + {n - n_train} dias (teste)")
    print(f"    Data de corte aproximada: {data_corte.date()}")

    return df, n_train


def calcular_metricas(df: pd.DataFrame, y_col: str, yhat_col: str, mask) -> dict:
    y = df.loc[mask, y_col]
    yhat = df.loc[mask, yhat_col]

    # remove casos sem previsão (NaN)
    valid = (~y.isna()) & (~yhat.isna())
    y = y[valid]
    yhat = yhat[valid]

    if len(y) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}

    erros = yhat - y
    mae = erros.abs().mean()
    rmse = (erros.pow(2).mean()) ** 0.5

    return {"MAE": mae, "RMSE": rmse, "n": len(y)}


def imprimir_metricas(nome_modelo: str, metr_train: dict, metr_test: dict):
    print(f"\nModelo: {nome_modelo}")
    print(f"  Treino: n={metr_train['n']}, MAE={metr_train['MAE']:.3f}, RMSE={metr_train['RMSE']:.3f}")
    print(f"  Teste : n={metr_test['n']}, MAE={metr_test['MAE']:.3f}, RMSE={metr_test['RMSE']:.3f}")


def salvar_previsoes(df: pd.DataFrame, caminho_csv: Path):
    print(f"\n[3] Salvando tabela de previsões em: {caminho_csv}")
    cols = [
        "data",
        "conjunto",
        "interrupcoes",
        "prev_persistencia",
        "prev_mm7",
    ]
    df[cols].to_csv(caminho_csv, index=False)


def plotar_previsoes_teste(df: pd.DataFrame, caminho_png: Path):
    print(f"[4] Gerando gráfico de previsões (período de teste): {caminho_png}")
    df = df[df["conjunto"] == "teste"].copy()
    df = df.dropna(subset=["prev_persistencia", "prev_mm7"])

    plt.figure(figsize=(12, 6))
    x = df["data"]

    plt.plot(x, df["interrupcoes"], label="Observado (interrupções)")
    plt.plot(x, df["prev_persistencia"], label="Prev. Persistência (y[t-1])")
    plt.plot(x, df["prev_mm7"], label="Prev. MM7 (média últimos 7 dias)")

    plt.xlabel("Data")
    plt.ylabel("Número de interrupções diárias")
    plt.title("Previsões diárias de interrupções (período de teste)")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300)
    plt.close()


# =====================================================
# 3) MAIN
# =====================================================

def main():
    print("ROOT_DIR =", ROOT_DIR)
    print("BASE_DIARIA_CSV =", BASE_DIARIA_CSV)
    print("SEGUNDO_PEDIDO_DIR =", SEGUNDO_PEDIDO_DIR)

    # 1) Carga base diária
    df = carregar_base_diaria(BASE_DIARIA_CSV)

    # 2) Criar previsões baselines (usando apenas dados passados)
    df = criar_previsoes_baseline(df)

    # 3) Divisão temporal treino x teste
    df, n_train = split_treino_teste(df, TRAIN_FRACTION)

    # 4) Métricas
    mask_treino = df["conjunto"] == "treino"
    mask_teste = df["conjunto"] == "teste"

    metr_pers_train = calcular_metricas(df, "interrupcoes", "prev_persistencia", mask_treino)
    metr_pers_test = calcular_metricas(df, "interrupcoes", "prev_persistencia", mask_teste)

    metr_mm7_train = calcular_metricas(df, "interrupcoes", "prev_mm7", mask_treino)
    metr_mm7_test = calcular_metricas(df, "interrupcoes", "prev_mm7", mask_teste)

    imprimir_metricas("Persistência (y[t-1])", metr_pers_train, metr_pers_test)
    imprimir_metricas("Média móvel 7 dias", metr_mm7_train, metr_mm7_test)

    # 5) Salvar tabela de previsões
    previsoes_csv = DADOS_DIR / "previsoes_diarias_baselines.csv"
    salvar_previsoes(df, previsoes_csv)

    # 6) Gráfico de observados x previstos (período de teste)
    grafico_png = GRAFICOS_DIR / "previsao_interrupcoes_baselines_teste.png"
    plotar_previsoes_teste(df, grafico_png)

    print("\n[OK] Tópico 4 concluído: previsão diária com divisão temporal treino/teste.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Algo deu errado:")
        print(e)
        sys.exit(1)
