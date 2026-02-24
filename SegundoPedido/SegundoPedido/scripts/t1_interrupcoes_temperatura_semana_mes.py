import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1) CONFIGURAÇÕES DE CAMINHO
# =====================================================

# Este arquivo está em:
#   /home/mateus/CLEAR DATA/SegundoPedido/SegundoPedido/scripts/t1_....py
THIS_FILE = Path(__file__).resolve()

# Raiz "CLEAR DATA"
ROOT_DIR = THIS_FILE.parents[3]   # -> /home/mateus/CLEAR DATA

# Projeto original de interrupções
INTERRUPCOES_DIR = ROOT_DIR / "interrupcoes-aneel"
INTERRUPCOES_CSV = INTERRUPCOES_DIR / "dados_completos_brasilia.csv"

# Arquivo diário de clima (ajuste o nome se for outro)
CLIMA_CSV = INTERRUPCOES_DIR / "clima_diario_brasilia.csv"

# Pasta da nova entrega (a "inner" SegundoPedido)
SEGUNDO_PEDIDO_DIR = THIS_FILE.parents[1]  # -> /home/mateus/CLEAR DATA/SegundoPedido/SegundoPedido

DADOS_DIR = SEGUNDO_PEDIDO_DIR / "dados"
GRAFICOS_DIR = SEGUNDO_PEDIDO_DIR / "graficos"

DADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2) CONFIGURAÇÕES DAS COLUNAS
# =====================================================

# Dados de interrupções (vêm de dados_completos_brasilia.csv)
COL_DATA_INICIO = "DatInicioInterrupcao"   # data/hora de início da interrupção
COL_ID_INTER = "NumOrdemInterrupcao"       # identificador da interrupção (iremos contar distintos por dia)

# Dados de clima (vêm de clima_diario_brasilia.csv) -> AJUSTAR CONFORME O SEU ARQUIVO
CLIMA_DATE_COL = "data"                    # coluna de data no arquivo de clima
CLIMA_TEMP_COL = "temperatura_media"       # temperatura média diária


# =====================================================
# 3) FUNÇÕES AUXILIARES
# =====================================================

def carregar_interrupcoes(caminho_csv: Path) -> pd.DataFrame:
    print(f"[1] Lendo interrupções de: {caminho_csv}")
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo de interrupções não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv)

    for col in [COL_DATA_INICIO, COL_ID_INTER]:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho_csv}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    # converte para datetime
    df[COL_DATA_INICIO] = pd.to_datetime(df[COL_DATA_INICIO])

    # cria uma coluna de data (somente dia) para agregação
    df["data"] = df[COL_DATA_INICIO].dt.date

    # conta interrupções distintas por dia (NumOrdemInterrupcao único)
    diario = (
        df.groupby("data")[COL_ID_INTER]
        .nunique()
        .reset_index()
        .rename(columns={COL_ID_INTER: "interrupcoes"})
    )

    # converte de volta para datetime (pandas) para facilitar resample depois
    diario["data"] = pd.to_datetime(diario["data"])

    print(f"[1] Tabela diária de interrupções gerada com {len(diario)} linhas.")
    return diario


def carregar_clima(caminho_csv: Path) -> pd.DataFrame:
    print(f"[2] Lendo clima diário de: {caminho_csv}")
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo de clima não encontrado: {caminho_csv}\n"
            f"Ajuste CLIMA_CSV ou crie esse arquivo."
        )

    df = pd.read_csv(caminho_csv)

    for col in [CLIMA_DATE_COL, CLIMA_TEMP_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho_csv}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    df[CLIMA_DATE_COL] = pd.to_datetime(df[CLIMA_DATE_COL])

    # renomeia para padronizar
    df = df.rename(columns={
        CLIMA_DATE_COL: "data",
        CLIMA_TEMP_COL: "temperatura_media"
    })

    print(f"[2] Tabela de clima diário carregada com {len(df)} linhas.")
    return df[["data", "temperatura_media"]]


def integrar_diario(interrupcoes_diario: pd.DataFrame, clima_diario: pd.DataFrame) -> pd.DataFrame:
    print("[3] Integrando interrupções diárias com clima diário...")
    base = pd.merge(
        interrupcoes_diario,
        clima_diario,
        on="data",
        how="inner"   # ou 'left' se quiser manter todos os dias de interrupção
    ).sort_values("data")

    print(f"[3] Base diária integrada com {len(base)} linhas.")
    return base


def agregar_por_frequencia(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq:
        'W'  -> semanal
        'MS' -> mensal (Month Start)
    """
    df = df.set_index("data")

    agg = df.resample(freq).agg(
        interrupcoes=("interrupcoes", "sum"),
        temperatura_media=("temperatura_media", "mean"),
    )

    agg = agg.dropna(subset=["temperatura_media"], how="all")
    agg = agg.reset_index().rename(columns={"data": "data_referencia"})

    return agg


def salvar_csv(df: pd.DataFrame, caminho: Path):
    print(f"    - Salvando CSV: {caminho}")
    df.to_csv(caminho, index=False)


def plotar_interrupcoes_temperatura(df: pd.DataFrame, titulo: str, arquivo_saida: Path):
    print(f"    - Gerando gráfico: {arquivo_saida}")
    plt.figure(figsize=(12, 6))

    x = df["data_referencia"]

    ax1 = plt.gca()
    ax1.plot(x, df["interrupcoes"], marker="o", label="Interrupções")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Número de interrupções")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, df["temperatura_media"], marker="s", linestyle="--", label="Temperatura média")
    ax2.set_ylabel("Temperatura média (°C)")

    plt.title(titulo)

    # legenda combinada
    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()


# =====================================================
# 4) ROTINA PRINCIPAL
# =====================================================

def main():
    print("ROOT_DIR =", ROOT_DIR)
    print("INTERRUPCOES_CSV =", INTERRUPCOES_CSV)
    print("CLIMA_CSV =", CLIMA_CSV)
    print("SEGUNDO_PEDIDO_DIR =", SEGUNDO_PEDIDO_DIR)

    # 1) Carregar interrupções e gerar base diária
    interrupcoes_diario = carregar_interrupcoes(INTERRUPCOES_CSV)

    # 2) Carregar clima diário
    clima_diario = carregar_clima(CLIMA_CSV)

    # 3) Integrar base diária
    base_diaria = integrar_diario(interrupcoes_diario, clima_diario)
    base_diaria_csv = DADOS_DIR / "base_diaria_interrupcoes_temperatura.csv"
    salvar_csv(base_diaria, base_diaria_csv)

    # 4) Agregação semanal
    print("[4] Agregando por semana...")
    semanal = agregar_por_frequencia(base_diaria, freq="W")
    semanal_csv = DADOS_DIR / "aggregados_semanal_interrupcoes_temperatura.csv"
    salvar_csv(semanal, semanal_csv)

    semanal_png = GRAFICOS_DIR / "interrupcoes_vs_temperatura_semanal.png"
    plotar_interrupcoes_temperatura(
        semanal,
        titulo="Interrupções x Temperatura (Agregado Semanal)",
        arquivo_saida=semanal_png,
    )

    # 5) Agregação mensal
    print("[5] Agregando por mês...")
    mensal = agregar_por_frequencia(base_diaria, freq="MS")
    mensal_csv = DADOS_DIR / "aggregados_mensal_interrupcoes_temperatura.csv"
    salvar_csv(mensal, mensal_csv)

    mensal_png = GRAFICOS_DIR / "interrupcoes_vs_temperatura_mensal.png"
    plotar_interrupcoes_temperatura(
        mensal,
        titulo="Interrupções x Temperatura (Agregado Mensal)",
        arquivo_saida=mensal_png,
    )

    print("\n[OK] Tópico 1 concluído: interrupções x temperatura (diário integrado, semanal e mensal).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Algo deu errado:")
        print(e)
        sys.exit(1)
