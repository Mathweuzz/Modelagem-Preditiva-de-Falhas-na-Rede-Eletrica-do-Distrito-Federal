"""
03 - Engenharia de Atributos (Feature Engineering)
===================================================

Constrói o dataset final usado pelos modelos preditivos
(`dataset_engenharia_features.csv`) a partir da base diária consolidada
INMET + ANEEL.

Operacoes aplicadas (todas determinísticas, ver Cap. 3 da monografia):

  0. Reindexacao para frequencia diaria estrita + interpolacao linear de
     variaveis meteorologicas (1 NaN em temperatura_media 03/01/2018).
     n_registros (variavel de completude da medicao, nao prospectiva) e removida.
  1. Features de calendario: mes, dia_semana, dia_ano
  2. Codificacao ciclica do mes: mes_sin, mes_cos
  3. Defasagens (lags) de 1, 2, 3 e 7 dias para:
       - interrupcoes, precipitacao_total_mm, temperatura_media, vento_rajada_max_ms
  4. Medias moveis exponenciais (EMA) de 3, 7 e 14 dias
  5. Desvio-padrao movel (rolling std) de 7 dias
  6. Drop de NaNs das primeiras 7 linhas (janela incompleta)

Fonte de dados:
  ../data/base_diaria_interrupcoes_clima_vento.csv

Saida:
  ../data/dataset_engenharia_features.csv

Execucao:
  cd Fonte/src && python 03_feature_engineering.py
"""
import pandas as pd
import numpy as np
import os

LAG_DAYS = [1, 2, 3, 7]
EMA_SPANS = [3, 7, 14]
ROLLING_STD_WINDOW = 7

LAG_COLUMNS = ['interrupcoes', 'precipitacao_total_mm',
               'temperatura_media', 'vento_rajada_max_ms']
EMA_COLUMNS = ['precipitacao_total_mm', 'temperatura_media', 'vento_rajada_max_ms']
STD_COLUMNS = ['precipitacao_total_mm', 'temperatura_media', 'vento_rajada_max_ms']

# Variaveis meteorologicas que podem ser interpoladas linearmente.
# interrupcoes NAO e interpolada: NaN de contagem nao tem valor fisico interpolavel.
METEO_COLUMNS = ['temperatura_media', 'precipitacao_total_mm',
                 'vento_velocidade_media_ms', 'vento_velocidade_max_ms',
                 'vento_rajada_max_ms', 'vento_dir_sin',
                 'vento_dir_cos']


def normalize_wind_direction_components(df):
    """Restaura vetores unitários após interpolar seno e cosseno."""
    columns = ['vento_dir_sin', 'vento_dir_cos']
    if not set(columns).issubset(df.columns):
        return df

    norm = np.hypot(df['vento_dir_sin'], df['vento_dir_cos'])
    invalid = norm < 1e-12
    if invalid.any():
        dates = [str(value.date()) for value in df.index[invalid][:5]]
        raise ValueError(
            'Direção do vento indefinida após interpolação; '
            f'exemplos de datas: {dates}'
        )
    df['vento_dir_sin'] = df['vento_dir_sin'] / norm
    df['vento_dir_cos'] = df['vento_dir_cos'] / norm
    return df


def load_base(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df = df.sort_index()
    return df


def fix_continuity(df):
    """
    Reindexar para frequencia diaria estrita e interpolar variaveis meteorologicas.
    Remove n_registros (variavel de completude da medicao, nao disponivel na previsao).
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df.index.name = 'data'

    # Remove coluna de completude de medicao
    if 'n_registros' in df.columns:
        df = df.drop(columns=['n_registros'])

    # Reconstrucao historica offline: a interpolacao linear usa pontos anteriores
    # e posteriores, portanto nao e causal. Em producao, usar imputacao baseada
    # somente nas observacoes disponiveis ate o instante corrente.
    # Interpolar apenas variaveis meteorologicas.
    meteo_present = [c for c in METEO_COLUMNS if c in df.columns]
    n_before = df[meteo_present].isnull().sum().sum()
    df[meteo_present] = df[meteo_present].interpolate(method='linear',
                                                      limit_direction='both')
    df = normalize_wind_direction_components(df)
    n_after = df[meteo_present].isnull().sum().sum()
    print(f"Interpolacao meteorologica: {n_before} NaN -> {n_after} NaN")

    # Verificacoes de integridade
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(days=1)]
    assert len(gaps) == 0, f"Ainda ha gaps temporais: {gaps}"

    nan_meteo = df[meteo_present].isnull().sum().sum()
    assert nan_meteo == 0, f"NaNs meteorologicos restantes: {nan_meteo}"

    return df


def add_calendar_features(df):
    df['mes'] = df.index.month
    df['dia_semana'] = df.index.dayofweek
    df['dia_ano'] = df.index.dayofyear
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    return df


def add_lags(df, columns, lags):
    for col in columns:
        for k in lags:
            df[f'{col}_lag_{k}'] = df[col].shift(k)
    return df


def add_emas(df, columns, spans):
    for col in columns:
        for s in spans:
            df[f'{col}_ema_{s}'] = df[col].ewm(span=s, adjust=False).mean()
    return df


def add_rolling_std(df, columns, window):
    for col in columns:
        df[f'{col}_std_{window}d'] = df[col].rolling(window=window).std()
    return df


def build_feature_dataset(filepath_in, filepath_out):
    df = load_base(filepath_in)
    print(f"Base de entrada: {df.shape[0]} dias, {df.shape[1]} colunas")

    df = fix_continuity(df)
    print(f"Apos reindexacao/interpolacao: {df.shape[0]} dias, {df.shape[1]} colunas")

    df = add_calendar_features(df)
    df = add_lags(df, LAG_COLUMNS, LAG_DAYS)
    df = add_emas(df, EMA_COLUMNS, EMA_SPANS)
    df = add_rolling_std(df, STD_COLUMNS, ROLLING_STD_WINDOW)

    df = df.dropna()
    print(f"Dataset final:  {df.shape[0]} dias, {df.shape[1]} colunas")

    # Verificacao final
    assert df.isnull().sum().sum() == 0, "NaNs restantes no dataset final!"
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(days=1)]
    assert len(gaps) == 0, f"Gaps temporais no dataset final: {gaps}"
    print("Verificacao OK: sem NaNs, sem gaps temporais.")

    df.to_csv(filepath_out)
    print(f"  -> {filepath_out}")


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)

    build_feature_dataset(
        filepath_in='../data/base_diaria_interrupcoes_clima_vento.csv',
        filepath_out='../data/dataset_engenharia_features.csv'
    )

    print("\n[OK] Engenharia de atributos concluida.")
