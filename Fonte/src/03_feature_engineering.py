"""
03 - Engenharia de Atributos (Feature Engineering)
===================================================

Constrói o dataset final usado pelos modelos preditivos
(`dataset_engenharia_features.csv`) a partir da base diária consolidada
INMET + ANEEL.

Operações aplicadas (todas determinísticas, ver Cap. 3 da monografia):

  1. Features de calendário: mes, dia_semana, dia_ano
  2. Codificação cíclica do mês: mes_sin = sin(2π·mes/12), mes_cos = cos(2π·mes/12)
  3. Defasagens (lags) de 1, 2, 3 e 7 dias para:
       - interrupcoes
       - precipitacao_total_mm
       - temperatura_media
       - vento_rajada_max_ms
  4. Médias móveis exponenciais (EMA) de 3, 7 e 14 dias para:
       - precipitacao_total_mm
       - temperatura_media
       - vento_rajada_max_ms
  5. Desvio-padrão móvel (rolling std) de 7 dias para as mesmas 3 variáveis
  6. Drop de NaNs (rows iniciais sem janela completa) -> 3.073 -> 3.059 dias

Fonte de dados:
  ../data/base_diaria_interrupcoes_clima_vento.csv

Saída:
  ../data/dataset_engenharia_features.csv  (mesmo arquivo já presente)

Execução:
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


def load_base(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df = df.sort_index()
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

    df = add_calendar_features(df)
    df = add_lags(df, LAG_COLUMNS, LAG_DAYS)
    df = add_emas(df, EMA_COLUMNS, EMA_SPANS)
    df = add_rolling_std(df, STD_COLUMNS, ROLLING_STD_WINDOW)

    df = df.dropna()
    print(f"Dataset final:  {df.shape[0]} dias, {df.shape[1]} colunas")

    df.to_csv(filepath_out)
    print(f"  -> {filepath_out}")


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)

    build_feature_dataset(
        filepath_in='../data/base_diaria_interrupcoes_clima_vento.csv',
        filepath_out='../data/dataset_engenharia_features.csv'
    )

    print("\n[OK] Engenharia de atributos concluída.")
