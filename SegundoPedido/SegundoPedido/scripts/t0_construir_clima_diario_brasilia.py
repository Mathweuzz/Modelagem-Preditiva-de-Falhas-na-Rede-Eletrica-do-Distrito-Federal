from pathlib import Path

import pandas as pd

# =====================================================
# 1) CONFIGURAÇÕES DE CAMINHO
# =====================================================

THIS_FILE = Path(__file__).resolve()

# /home/mateus/CLEAR DATA
ROOT_DIR = THIS_FILE.parents[3]

# Pasta com os CSVs horários do INMET
CLIMA_HOURLY_DIR = ROOT_DIR / "dados_clima-inmet_limpos"

# Projeto original de interrupções
INTERRUPCOES_DIR = ROOT_DIR / "interrupcoes-aneel"

# Arquivo de saída: clima diário consolidado
CLIMA_DIARIO_CSV = INTERRUPCOES_DIR / "clima_diario_brasilia.csv"

# Filtrar apenas estação de Brasília A001
STATION_SUBSTRING = "A001_BRASILIA"  # aparece no nome do arquivo


# =====================================================
# 2) FUNÇÕES
# =====================================================

def listar_arquivos_estacao(base_dir: Path, station_substring: str):
    arquivos = []
    for path in base_dir.rglob("*.CSV"):
        if station_substring in path.name:
            arquivos.append(path)
    arquivos = sorted(arquivos)
    print(f"Encontrados {len(arquivos)} arquivos para a estação '{station_substring}':")
    for p in arquivos:
        print("  -", p)
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em {base_dir} contendo '{station_substring}' no nome."
        )
    return arquivos


def processar_arquivo_horario(caminho: Path) -> pd.DataFrame:
    print(f"Lendo arquivo horário: {caminho}")
    df = pd.read_csv(
        caminho,
        na_values=[-9999, -9999.0],
    )

    # Conferir colunas necessárias
    obrigatorias = [
        "data",
        "precipitacao_total_horario_mm",
        "temperatura_ar_bulbo_seco_c",
    ]
    for col in obrigatorias:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada em {caminho}.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    # Converter data
    df["data"] = pd.to_datetime(df["data"])

    # Agregação diária
    diario = (
        df.groupby("data")
        .agg(
            precipitacao_total_mm=("precipitacao_total_horario_mm", "sum"),
            temperatura_media=("temperatura_ar_bulbo_seco_c", "mean"),
            temperatura_max=("temperatura_ar_bulbo_seco_c", "max"),
            temperatura_min=("temperatura_ar_bulbo_seco_c", "min"),
        )
        .reset_index()
    )

    return diario


def construir_clima_diario():
    arquivos = listar_arquivos_estacao(CLIMA_HOURLY_DIR, STATION_SUBSTRING)

    todos_anos = []
    for arq in arquivos:
        diario = processar_arquivo_horario(arq)
        todos_anos.append(diario)

    clima_diario = pd.concat(todos_anos, ignore_index=True)

    # Remover dias totalmente vazios (se houver)
    clima_diario = clima_diario.sort_values("data")
    clima_diario = clima_diario.dropna(
        subset=["temperatura_media", "precipitacao_total_mm"], how="all"
    ).reset_index(drop=True)

    print(f"\nTotal de linhas no clima diário consolidado: {len(clima_diario)}")
    print(f"Primeira data: {clima_diario['data'].min()}")
    print(f"Última data: {clima_diario['data'].max()}")

    print(f"\nSalvando arquivo diário em: {CLIMA_DIARIO_CSV}")
    clima_diario.to_csv(CLIMA_DIARIO_CSV, index=False)


# =====================================================
# 3) MAIN
# =====================================================

if __name__ == "__main__":
    construir_clima_diario()
