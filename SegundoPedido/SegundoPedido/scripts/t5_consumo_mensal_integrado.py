import sys
from pathlib import Path

import pandas as pd

# =====================================================
# 1) CAMINHOS
# =====================================================

THIS_FILE = Path(__file__).resolve()

# /home/mateus/CLEAR DATA
ROOT_DIR = THIS_FILE.parents[3]

# Pasta da nova entrega (inner SegundoPedido)
SEGUNDO_PEDIDO_DIR = THIS_FILE.parents[1]
DADOS_DIR = SEGUNDO_PEDIDO_DIR / "dados"

DADOS_DIR.mkdir(parents=True, exist_ok=True)

# Base diária integrada (interrupções + clima), criada no t3
BASE_DIARIA_CLIMA_CSV = DADOS_DIR / "base_diaria_interrupcoes_clima.csv"

# Arquivo de balanço mensal (samp-aneel)
SAMP_BALANCO_CSV = ROOT_DIR / "samp-aneel" / "samp-balanco.csv"

# Arquivo de saída com consumo mensal
CONSUMO_MENSAL_CSV = ROOT_DIR / "interrupcoes-aneel" / "consumo_mensal_brasilia.csv"

# Arquivo de saída: base mensal completa (interrupções + clima + consumo)
BASE_MENSAL_COMPLETA_CSV = DADOS_DIR / "base_mensal_interrupcoes_clima_consumo.csv"


# =====================================================
# 2) CONFIGURAÇÕES DE COLUNAS
# =====================================================

# Campos no samp-balanco.csv
COL_CNPJ = "NumCPFCNPJ"
COL_MODALIDADE = "DscModalidadeBalanco"
COL_FLUXO = "DscFluxoEnergia"
COL_CCT = "DscCctBalanco"
COL_DETALHE = "DscDetalheBalanco"
COL_ANO = "AnoReferenciaBalanco"
COL_MES = "MesReferenciaBalanco"
COL_VLR = "VlrEnergia"

# CNPJ Neoenergia Brasília (com zero à esquerda)
CNPJ_NEO_DF = "07522669000192"

# Filtro que vamos usar como “consumo total do mês”
VAL_MODALIDADE = "Energia Injetada"
VAL_FLUXO = "Disponibilidades"
VAL_CCT = "Energia Injetada Total"
VAL_DETALHE = "Energia Medida (kWh)"  # só o total geral, sem faixa de tensão

# =====================================================
# 3) FUNÇÕES
# =====================================================

def carregar_base_diaria_clima(caminho: Path) -> pd.DataFrame:
    print(f"[1] Lendo base diária de interrupções + clima: {caminho}")
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho)
    if "data" not in df.columns:
        raise ValueError(
            f"Coluna 'data' não encontrada em {caminho}.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    print(f"[1] Base diária com {len(df)} linhas.")
    return df


def agregar_mensal_interrupcoes_clima(df: pd.DataFrame) -> pd.DataFrame:
    print("[2] Agregando interrupções + clima para base mensal...")
    df = df.set_index("data")

    mensal = df.resample("MS").agg(
        interrupcoes=("interrupcoes", "sum"),
        temperatura_media=("temperatura_media", "mean"),
        precipitacao_total_mm=("precipitacao_total_mm", "sum"),
    )

    mensal = mensal.reset_index().rename(columns={"data": "data_referencia"})
    print(f"[2] Base mensal (interrupções + clima) com {len(mensal)} linhas.")
    return mensal


def extrair_consumo_mensal(caminho: Path) -> pd.DataFrame:
    print(f"[3] Lendo balanço mensal (samp-aneel): {caminho}")
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    # ATENÇÃO: engine='python' e on_bad_lines='skip' para ignorar linhas com vírgula quebrando campos
    df = pd.read_csv(
        caminho,
        dtype={COL_CNPJ: str},
        engine="python",
        on_bad_lines="skip",
    )

    obrigatorias = [
        COL_CNPJ, COL_MODALIDADE, COL_FLUXO, COL_CCT,
        COL_DETALHE, COL_ANO, COL_MES, COL_VLR
    ]
    for col in obrigatorias:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    # Filtro para Neoenergia Brasília + Energia Injetada Total medida em kWh (total geral)
    mask = (
        (df[COL_CNPJ] == CNPJ_NEO_DF) &
        (df[COL_MODALIDADE] == VAL_MODALIDADE) &
        (df[COL_FLUXO] == VAL_FLUXO) &
        (df[COL_CCT] == VAL_CCT) &
        (df[COL_DETALHE] == VAL_DETALHE)
    )
    filtrado = df.loc[mask].copy()

    if filtrado.empty:
        raise ValueError(
            "Nenhuma linha encontrada em samp-balanco com o filtro definido.\n"
            "Verifique se os textos de DscModalidadeBalanco, DscFluxoEnergia, "
            "DscCctBalanco e DscDetalheBalanco batem exatamente com o arquivo."
        )

    print(f"[3] Linhas após filtro (Neoenergia DF + Energia Injetada Total, kWh): {len(filtrado)}")

    filtrado[COL_ANO] = filtrado[COL_ANO].astype(int)
    filtrado[COL_MES] = filtrado[COL_MES].astype(int)
    filtrado[COL_VLR] = filtrado[COL_VLR].astype(float)

    consumo_mensal = (
        filtrado
        .groupby([COL_ANO, COL_MES])[COL_VLR]
        .sum()
        .reset_index()
        .rename(columns={COL_VLR: "consumo_total_kwh"})
    )

    consumo_mensal["data_referencia"] = pd.to_datetime({
        "year": consumo_mensal[COL_ANO],
        "month": consumo_mensal[COL_MES],
        "day": 1,
    })

    consumo_mensal = consumo_mensal.sort_values("data_referencia").reset_index(drop=True)
    print(
        f"[3] Consumo mensal com {len(consumo_mensal)} linhas "
        f"(de {consumo_mensal['data_referencia'].min().date()} "
        f"até {consumo_mensal['data_referencia'].max().date()})."
    )

    print(f"[3] Salvando consumo mensal em: {CONSUMO_MENSAL_CSV}")
    consumo_mensal.to_csv(CONSUMO_MENSAL_CSV, index=False)

    return consumo_mensal[["data_referencia", "consumo_total_kwh"]]


def integrar_mensal(interrup_clima_mensal: pd.DataFrame, consumo_mensal: pd.DataFrame) -> pd.DataFrame:
    print("[4] Integrando base mensal (interrupções + clima) com consumo mensal...")
    base = pd.merge(
        interrup_clima_mensal,
        consumo_mensal,
        on="data_referencia",
        how="inner"
    ).sort_values("data_referencia")

    print(f"[4] Base mensal completa com {len(base)} linhas.")
    return base


# =====================================================
# 4) MAIN
# =====================================================

def main():
    print("ROOT_DIR =", ROOT_DIR)
    print("BASE_DIARIA_CLIMA_CSV =", BASE_DIARIA_CLIMA_CSV)
    print("SAMP_BALANCO_CSV =", SAMP_BALANCO_CSV)
    print("CONSUMO_MENSAL_CSV =", CONSUMO_MENSAL_CSV)
    print("BASE_MENSAL_COMPLETA_CSV =", BASE_MENSAL_COMPLETA_CSV)

    base_diaria = carregar_base_diaria_clima(BASE_DIARIA_CLIMA_CSV)
    base_mensal_int_clima = agregar_mensal_interrupcoes_clima(base_diaria)
    consumo_mensal = extrair_consumo_mensal(SAMP_BALANCO_CSV)
    base_mensal_completa = integrar_mensal(base_mensal_int_clima, consumo_mensal)

    print(f"[5] Salvando base mensal completa em: {BASE_MENSAL_COMPLETA_CSV}")
    base_mensal_completa.to_csv(BASE_MENSAL_COMPLETA_CSV, index=False)

    print("\n[OK] Tópico 5 concluído: base mensal com interrupções, clima e consumo.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Algo deu errado:")
        print(e)
        sys.exit(1)