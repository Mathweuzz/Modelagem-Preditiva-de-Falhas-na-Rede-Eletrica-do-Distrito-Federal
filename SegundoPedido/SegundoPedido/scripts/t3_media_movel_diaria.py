import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1) CONFIGURAÇÕES DE CAMINHO
# =====================================================

THIS_FILE = Path(__file__).resolve()

# /home/mateus/CLEAR DATA
ROOT_DIR = THIS_FILE.parents[3]

# Projeto original de interrupções
INTERRUPCOES_DIR = ROOT_DIR / "interrupcoes-aneel"
INTERRUPCOES_CSV = INTERRUPCOES_DIR / "dados_completos_brasilia.csv"

# Clima diário consolidado (já criado no t0)
CLIMA_CSV = INTERRUPCOES_DIR / "clima_diario_brasilia.csv"

# Pasta da nova entrega (inner SegundoPedido)
SEGUNDO_PEDIDO_DIR = THIS_FILE.parents[1]
DADOS_DIR = SEGUNDO_PEDIDO_DIR / "dados"
GRAFICOS_DIR = SEGUNDO_PEDIDO_DIR / "graficos"

DADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2) CONFIGURAÇÕES DAS COLUNAS
# =====================================================

# Interrupções
COL_DATA_INICIO = "DatInicioInterrupcao"
COL_ID_INTER = "NumOrdemInterrupcao"

# Clima diário
CLIMA_DATE_COL = "data"
CLIMA_TEMP_COL = "temperatura_media"
CLIMA_PRECIP_COL = "precipitacao_total_mm"

# Janelas de média móvel (em dias)
MM_WINDOWS = [7, 14]


# =====================================================
# 3) FUNÇÕES AUXILIARES
# =====================================================

def carregar_interrupcoes_diarias(caminho_csv: Path) -> pd.DataFrame:
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

    df[COL_DATA_INICIO] = pd.to_datetime(df[COL_DATA_INICIO])
    df["data"] = df[COL_DATA_INICIO].dt.date

    diario = (
        df.groupby("data")[COL_ID_INTER]
        .nunique()
        .reset_index()
        .rename(columns={COL_ID_INTER: "interrupcoes"})
    )
    diario["data"] = pd.to_datetime(diario["data"])

    print(f"[1] Tabela diária de interrupções com {len(diario)} linhas.")
    return diario


def carregar_clima_diario(caminho_csv: Path) -> pd.DataFrame:
    print(f"[2] Lendo clima diário de: {caminho_csv}")
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo de clima não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv)

    for col in [CLIMA_DATE_COL, CLIMA_TEMP_COL, CLIMA_PRECIP_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho_csv}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    df[CLIMA_DATE_COL] = pd.to_datetime(df[CLIMA_DATE_COL])

    df = df.rename(columns={
        CLIMA_DATE_COL: "data",
        CLIMA_TEMP_COL: "temperatura_media",
        CLIMA_PRECIP_COL: "precipitacao_total_mm",
    })

    print(f"[2] Clima diário com {len(df)} linhas.")
    return df[["data", "temperatura_media", "precipitacao_total_mm"]]


def integrar_base_diaria(interrupcoes: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    print("[3] Integrando interrupções + clima (diário)...")
    base = pd.merge(
        interrupcoes,
        clima,
        on="data",
        how="inner"
    ).sort_values("data")

    print(f"[3] Base diária integrada com {len(base)} linhas.")
    return base


def adicionar_medias_moveis(df: pd.DataFrame, col: str, windows) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("data")
    for w in windows:
        mm_col = f"{col}_mm_{w}"
        df[mm_col] = df[col].rolling(window=w, min_periods=1, center=False).mean()
        print(f"    - Criada coluna de média móvel: {mm_col}")
    return df


def plotar_serie_com_mm(df: pd.DataFrame, col_original: str, titulo: str, arquivo_saida: Path):
    print(f"    - Gerando gráfico: {arquivo_saida}")
    plt.figure(figsize=(12, 6))
    x = df["data"]

    # série original
    plt.plot(x, df[col_original], label=col_original)

    # médias móveis (todas as colunas que começam com col_original + '_mm_')
    for col in df.columns:
        prefix = f"{col_original}_mm_"
        if col.startswith(prefix):
            plt.plot(x, df[col], label=col)

    plt.xlabel("Data")
    plt.ylabel(col_original)
    plt.title(titulo)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()


# =====================================================
# 4) MAIN
# =====================================================

def main():
    print("ROOT_DIR =", ROOT_DIR)
    print("INTERRUPCOES_CSV =", INTERRUPCOES_CSV)
    print("CLIMA_CSV =", CLIMA_CSV)
    print("SEGUNDO_PEDIDO_DIR =", SEGUNDO_PEDIDO_DIR)

    # 1) Carga e integração diária
    interrupcoes_diario = carregar_interrupcoes_diarias(INTERRUPCOES_CSV)
    clima_diario = carregar_clima_diario(CLIMA_CSV)
    base_diaria = integrar_base_diaria(interrupcoes_diario, clima_diario)

    # salvar base diária "crua"
    base_diaria_path = DADOS_DIR / "base_diaria_interrupcoes_clima.csv"
    print(f"[4] Salvando base diária integrada em: {base_diaria_path}")
    base_diaria.to_csv(base_diaria_path, index=False)

    # 2) Médias móveis para cada variável
    print("[5] Calculando médias móveis (7 e 14 dias)...")
    df_mm = base_diaria.copy()

    for col in ["interrupcoes", "temperatura_media", "precipitacao_total_mm"]:
        print(f"  * Variável: {col}")
        df_mm = adicionar_medias_moveis(df_mm, col, MM_WINDOWS)

    # salvar base com médias móveis
    base_mm_path = DADOS_DIR / "base_diaria_interrupcoes_clima_mm.csv"
    print(f"[5] Salvando base com médias móveis em: {base_mm_path}")
    df_mm.to_csv(base_mm_path, index=False)

    # 3) Gráficos
    print("[6] Gerando gráficos diários com médias móveis...")

    # interrupções
    plotar_serie_com_mm(
        df_mm,
        col_original="interrupcoes",
        titulo="Interrupções diárias com médias móveis (7 e 14 dias)",
        arquivo_saida=GRAFICOS_DIR / "diario_interrupcoes_medias_moveis.png",
    )

    # temperatura média
    plotar_serie_com_mm(
        df_mm,
        col_original="temperatura_media",
        titulo="Temperatura média diária com médias móveis (7 e 14 dias)",
        arquivo_saida=GRAFICOS_DIR / "diario_temperatura_medias_moveis.png",
    )

    # precipitação
    plotar_serie_com_mm(
        df_mm,
        col_original="precipitacao_total_mm",
        titulo="Precipitação diária com médias móveis (7 e 14 dias)",
        arquivo_saida=GRAFICOS_DIR / "diario_precipitacao_medias_moveis.png",
    )

    print("\n[OK] Tópico 3 concluído: médias móveis diárias para interrupções, temperatura e precipitação.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Algo deu errado:")
        print(e)
        sys.exit(1)
